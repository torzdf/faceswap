#!/usr/bin/env python3
""" Rich interactive preview-display backend that renders training frames into a window as either a
standalone viewer or an embedded panel, chosen at runtime in preference to the OpenCV fallback

This module implements the Tkinter side of ``PreviewBase``: it builds a taskbar (scale slider,
interpolation radios and an optional Save button), a scrollable canvas with fit/auto-scale
behaviour and a cached image wrapper, then wires up mouse and keyboard bindings before running the
inherited ``_launch`` loop that waits for new previews from ``PreviewBuffer`` and renders each on
demand. The viewer can run alone as a fullscreen window (its real size is captured while iconified
so it can be restored) or embedded inside another GUI when handed a parent frame, in which case the
standalone controls are dropped.
"""
from __future__ import annotations
import logging
import os
import sys
import tkinter as tk
import typing as T

from datetime import datetime
from platform import system
from tkinter import ttk
from math import ceil, floor

from PIL import Image, ImageTk

import cv2

from lib.logger import parse_class_init
from lib.utils import get_module_objects

from .preview import PreviewBase

if T.TYPE_CHECKING:
    import numpy as np
    from .preview import PreviewBuffer, TriggerKeysType, TriggerType

logger = logging.getLogger(__name__)


class _Taskbar():
    """ Construct the standalone preview option bar (scale combo/slider, interpolation radios and
    an optional Save button)

    Holds each widget's backing variable so the canvas can react to scale/interpolation changes
    while running. It is built once by `PreviewTk` and exists only in standalone mode; embedded
    bars ride inside a caller-supplied frame without those controls

    Parameters
    ----------
    parent
        The widget this bar should be packed into when building its own layout (standalone use)
    taskbar
        An existing ``ttk.Frame`` to build within for embedded use; ``None`` marks standalone mode
        so the bar builds and self-packs its frame at the window bottom

    Notes
    -----
    The Save button is only added in standalone viewers, where the bar also packs itself into a
    dedicated row below the canvas
    """
    def __init__(self, parent: tk.Frame, taskbar: ttk.Frame | None) -> None:
        logger.debug(parse_class_init(locals()))
        self._is_standalone = taskbar is None
        self._gui_mapped: list[tk.Widget] = []
        self._frame = tk.Frame(parent) if taskbar is None else taskbar
        self._min_max_scales = (20, 400)
        self._vars = {"save": tk.BooleanVar(),
                      "scale": tk.StringVar(),
                      "slider": tk.IntVar(),
                      "interpolator": tk.IntVar()}
        self._interpolators = [("nearest_neighbour", cv2.INTER_NEAREST),
                               ("bicubic", cv2.INTER_CUBIC)]
        self._scale = self._add_scale_combo()
        self._slider = self._add_scale_slider()
        self._add_interpolator_radio()
        if self._is_standalone:
            self._add_save_button()
            self._frame.pack(side=tk.BOTTOM, fill=tk.X, padx=2, pady=2)

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        return (f"{self.__class__.__name__}("
                f"parent={self._frame.winfo_parent()!r}, "
                f"taskbar={None if self._is_standalone else self._frame!r})")

    @property
    def min_scale(self) -> int:
        """ The lower scale percentage bound for the slider and combo box """
        return self._min_max_scales[0]

    @property
    def max_scale(self) -> int:
        """ The upper scale percentage bound for the slider and combo box """
        return self._min_max_scales[1]

    @property
    def save_var(self) -> tk.BooleanVar:
        """ The ``BooleanVar`` flag that requests a preview save when checked """
        retval = self._vars["save"]
        assert isinstance(retval, tk.BooleanVar)
        return retval

    @property
    def scale_var(self) -> tk.StringVar:
        """ The ``StringVar`` holding the current display scale as text (eg: ``"100%"``) """
        retval = self._vars["scale"]
        assert isinstance(retval, tk.StringVar)
        return retval

    @property
    def slider_var(self) -> tk.IntVar:
        """ The ``IntVar`` tracking the numeric scale value for the slider """
        retval = self._vars["slider"]
        assert isinstance(retval, tk.IntVar)
        return retval

    @property
    def interpolator_var(self) -> tk.IntVar:
        """ The ``IntVar`` for the selected resampling-mode index (nearest-neighbour/bicubic) """
        retval = self._vars["interpolator"]
        assert isinstance(retval, tk.IntVar)
        return retval

    def _track_widget(self, widget: tk.Widget) -> None:
        """ Append a gui related widget to the tracked list for later teardown

        Parameters
        ----------
        widget
            The option-bar widget to track for later teardown
        """
        if self._is_standalone:
            return
        logger.debug("[_Tasbar] Tracking option bar widget for GUI: %s", widget)
        self._gui_mapped.append(widget)

    def _add_scale_combo(self) -> ttk.Combobox:
        """ Add a read-only scale combo box whose choices are populated by `set_min_max_scale`

        Returns
        -------
        The created scale-combo box widget
        """
        logger.debug("[_Tasbar] Adding scale combo")
        self.scale_var.set("100%")
        scale = ttk.Combobox(self._frame,
                             textvariable=self.scale_var,
                             values=["Fit"],
                             state="readonly",
                             width=10)
        scale.pack(side=tk.RIGHT)
        scale.bind("<FocusIn>", self._clear_combo_focus)  # Remove auto-focus on widget text box
        self._track_widget(scale)
        logger.debug("[_Tasbar] Added scale combo: '%s'", scale)
        return scale

    def _clear_combo_focus(self, *args) -> None:  # pylint:disable=unused-argument
        """ Remove auto-focus from the scale combo so typing in it does not steal focus """
        logger.debug("[_Tasbar] Clearing scale combo focus")
        self._scale.selection_clear()
        self._scale.winfo_toplevel().focus_set()
        logger.debug("[_Tasbar] Cleared scale combo focus")

    def _add_scale_slider(self) -> tk.Scale:
        """ Add a horizontal slider bound to `slider_var` that drives zoom updates

        Returns
        -------
        The bound percentage zoom slider
        """
        logger.debug("[_Tasbar] Adding scale slider")
        self.slider_var.set(100)
        slider = tk.Scale(self._frame,
                          orient=tk.HORIZONTAL,
                          to=self.max_scale,
                          showvalue=False,
                          variable=self.slider_var,
                          command=self._on_slider_update)
        slider.pack(side=tk.RIGHT)
        self._track_widget(slider)
        logger.debug("[_Tasbar] Added scale slider: '%s'", slider)
        return slider

    def _add_interpolator_radio(self) -> None:
        """ Add radio buttons cycling between nearest-neighbour and bicubic resampling modes """
        frame = tk.Frame(self._frame)
        for text, mode in self._interpolators:
            logger.debug("[_Tasbar] Adding %s radio button", text)
            radio = tk.Radiobutton(frame, text=text, value=mode, variable=self.interpolator_var)
            radio.pack(side=tk.LEFT, anchor=tk.W)
            self._track_widget(radio)
            logger.debug("[_Tasbar] Added %s radio button", radio)
        self.interpolator_var.set(cv2.INTER_NEAREST)
        frame.pack(side=tk.RIGHT)
        self._track_widget(frame)

    def _add_save_button(self) -> None:
        """ Add a Save button whose command flips the `save_var` flag so previews are written """
        logger.debug("[_Tasbar] Adding save button")
        button = tk.Button(self._frame,
                           text="Save",
                           cursor="hand2",
                           command=lambda: self.save_var.set(True))
        button.pack(side=tk.LEFT)
        logger.debug("[_Tasbar] Added save button: '%s'", button)

    def _on_slider_update(self, value) -> None:
        """ Set the scale combo text from the slider position so both controls stay in sync

        Parameters
        ----------
        value
            The integer percentage (for example ``150``) read from the slider at update time
        """
        self.scale_var.set(f"{value}%")

    def set_min_max_scale(self, min_scale: int, max_scale: int) -> None:
        """ Rebuild the scale choices and bounds from new limits so the combo/slider reflect them

        Parameters
        ----------
        min_scale
            The smallest percentage the slider is allowed to reach.
        max_scale
            The largest percentage the slider is allowed to reach.
        """
        logger.debug("[_Tasbar] Setting min/max scales: (min: %s, max: %s)", min_scale, max_scale)
        self._min_max_scales = (min_scale, max_scale)
        self._slider.config(from_=self.min_scale, to=max_scale)
        scales = [10, 25, 50, 75, 100, 200, 300, 400, 800]
        if min_scale not in scales:
            scales.insert(0, min_scale)
        if max_scale not in scales:
            scales.append(max_scale)
        choices = ["Fit", *[f"{x}%" for x in scales if self.max_scale >= x >= self.min_scale]]
        self._scale.config(values=choices)
        logger.debug("[_Tasbar] Set min/max scale. min_max_scales: %s, scale combo choices: %s",
                     self._min_max_scales, choices)

    def cycle_interpolators(self, *args) -> None:  # pylint:disable=unused-argument
        """ Advance to the next resampling mode and select it via `interpolator_var` """
        current = next(i for i in self._interpolators if i[1] == self.interpolator_var.get())
        next_idx = self._interpolators.index(current) + 1
        next_idx = 0 if next_idx == len(self._interpolators) else next_idx
        self.interpolator_var.set(self._interpolators[next_idx][1])

    def destroy_widgets(self) -> None:
        """ Unpack, destroy and drop every tracked GUI widget so it can be garbage collected """
        if self._is_standalone:
            return
        for widget in reversed(self._gui_mapped):
            try:
                if not widget.winfo_exists():
                    continue
                if widget.winfo_ismapped():
                    logger.debug("[_Tasbar] Removing widget: %s", widget)
                    widget.pack_forget()
                    widget.destroy()
                    del widget
            except tk.TclError:
                continue
        self._gui_mapped.clear()
        for var in list(self._vars):
            logger.debug("[_Tasbar] Deleting tk variable: %s", var)
            del self._vars[var]


class _PreviewCanvas(tk.Canvas):  # pylint:disable=too-many-ancestors
    """ Render a scrollable preview frame on a canvas that resizes and centers its image

    The canvas wraps a ``tk.Canvas`` in an extra frame so scrollbars can be layered around it,
    binds to window-resize events to keep the image centered, and exposes the backing photo for the
    parent GUI to read when embedded

    Parameters
    ----------
    parent
        The widget this canvas should be packed into (the master frame supplied by `PreviewTk`)
    scale_var
        The ``StringVar`` bound to the scale controls so switching between Fit and a percentage re-
        renders
    screen_dimensions
        The outer GUI size used as a ceiling when the image is moved toward the top/left edge at
        startup
    is_standalone
        Whether this canvas owns its sizing; embedded canvases must not rescale themselves to their
        photo

    Notes
    -----
    In standalone mode the canvas sizes itself to the source image so it fills the window; embedded
    panels stay fixed and scroll instead.
    """
    def __init__(self,
                 parent: tk.Frame,
                 scale_var: tk.StringVar,
                 screen_dimensions: tuple[int, int],
                 is_standalone: bool) -> None:
        logger.debug(parse_class_init(locals()))
        frame = tk.Frame(parent)
        super().__init__(frame)
        self._parent = parent
        self._is_standalone = is_standalone
        self._screen_dimensions = screen_dimensions
        self._var_scale = scale_var
        self._configure_scrollbars(frame)
        self._image: ImageTk.PhotoImage | None = None
        self._image_id = self.create_image(self.width / 2,
                                           self.height / 2,
                                           anchor=tk.CENTER,
                                           image=self._image)
        self.pack(fill=tk.BOTH, expand=True)
        self.bind("<Configure>", self._resize)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        return (f"{self.__class__.__name__}("
                f"parent={self._parent!r}, "
                f"scale_var={self._var_scale!r}, "
                f"screen_dimensions={self._screen_dimensions!r}, "
                f"is_standalone={self._is_standalone!r})")

    @property
    def image_id(self) -> int:
        """ The id of the image item drawn on the canvas (used for zoom/pan bindings) """
        return self._image_id

    @property
    def width(self) -> int:
        """ The current mapped width of the canvas in pixels """
        return self.winfo_width()

    @property
    def height(self) -> int:
        """ The current mapped height of the canvas in pixels """
        return self.winfo_height()

    def _configure_scrollbars(self, frame: tk.Frame) -> None:
        """ Add horizontal and vertical scrollbars around the canvas and wire them to its views

        Parameters
        ----------
        frame
            The parent frame containing the canvas and its scrollbars
        """
        logger.debug("[_PreviewCanvas] Configuring scrollbars")
        x_scrollbar = tk.Scrollbar(frame, orient="horizontal", command=self.xview)
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        y_scrollbar = tk.Scrollbar(frame, command=self.yview)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.configure(xscrollcommand=x_scrollbar.set, yscrollcommand=y_scrollbar.set)
        logger.debug("[_PreviewCanvas] Configured scrollbars. x: '%s', y: '%s'",
                     x_scrollbar, y_scrollbar)

    def _resize(self, event: tk.Event) -> None:  # pylint:disable=unused-argument
        """ Reset scroll region and re-center the image after a window resize (or Fit request) """
        if self._var_scale.get() == "Fit":  # Trigger an update to resize image
            logger.debug("[_PreviewCanvas] Triggering redraw for 'Fit' Scaling")
            self._var_scale.set("Fit")
            return
        self.configure(scrollregion=self.bbox("all"))
        self.update_idletasks()
        assert self._image is not None
        self._center_image(self.width / 2, self.height / 2)
        # Move to top left when resizing into screen dimensions (initial startup)
        if self.width > self._screen_dimensions[0]:
            logger.debug("[_PreviewCanvas] Moving image to left edge")
            self.xview_moveto(0.0)
        if self.height > self._screen_dimensions[1]:
            logger.debug("[_PreviewCanvas] Moving image to top edge")
            self.yview_moveto(0.0)

    def _center_image(self, point_x: float, point_y: float) -> None:
        """ Move the drawn image to an anchor point so it stays centered within the canvas

        Parameters
        ----------
        point_x
            X coordinate of the anchor point
        point_y
            Y coordinate of the anchor point
        """
        canvas_location = (self.canvasx(point_x), self.canvasy(point_y))
        logger.debug("[_PreviewCanvas] Centering canvas for size (%s, %s). New image "
                     "coordinates: %s",
                     point_x, point_y, canvas_location)
        self.coords(self.image_id, canvas_location)

    def set_image(self,
                  image: ImageTk.PhotoImage,
                  center_image: bool = False) -> None:
        """ Display a photo on the canvas and size it to fill the window when standalone

        Parameters
        ----------
        image
            The ``ImageTk.PhotoImage`` to draw
        center_image, optional
            Whether to move the image to the canvas center after placing it; skipped in standalone
            mode where the canvas already fills its frame. Default: ``False``
        """
        logger.debug("[_PreviewCanvas] Setting canvas image. ID: %s, size: %s for canvas size: %s "
                     "(recenter: %s)",
                     self.image_id, (image.width(), image.height()), (self.width, self.height),
                     center_image)
        self._image = image
        self.itemconfig(self.image_id, image=self._image)
        if self._is_standalone:  # canvas size should not be updated inside GUI
            self.config(width=self._image.width(), height=self._image.height())
        self.update_idletasks()
        if center_image:
            self._center_image(self.width / 2, self.height / 2)
        self.configure(scrollregion=self.bbox("all"))
        logger.debug("[_PreviewCanvas] set canvas image. Canvas size: %s",
                     (self.width, self.height))


class _Image():
    """ Convert stored preview frames from BGR numpy arrays into scaled RGB ``PhotoImages`` for
    display

    This helper owns no window logic: it keeps the source frame plus a scale factor and resampling
    filter, then rebuilds a PIL ``PhotoImage`` on demand (resizing with nearest-neighbour when
    zoomed out and the chosen interpolator when zoomed in). It also writes PNG previews when asked
    to save

    Parameters
    ----------
    save_variable
        The ``BooleanVar`` flag that requests a preview be written to disk whenever it changes
    is_standalone
        Whether saves should target the script's directory (standalone) or a caller-supplied path
        (embedded)

    Notes
    -----
    The scale starts at 100%; only when scaled away from full resolution does the frame get resized
    before display
    """
    def __init__(self, save_variable: tk.BooleanVar, is_standalone: bool) -> None:
        logger.debug(parse_class_init(locals()))
        self._is_standalone = is_standalone
        self._source: np.ndarray | None = None
        self._display: ImageTk.PhotoImage | None = None
        self._scale = 1.0
        self._interpolation = cv2.INTER_NEAREST
        self._save_var = save_variable
        self._save_var.trace("w", self.save_preview)

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        return (f"{self.__class__.__name__}("
                f"save_variable={self._save_var!r}, "
                f"is_standalone={self._is_standalone!r})")

    @property
    def display_image(self) -> ImageTk.PhotoImage:
        """ The currently built RGB ``PhotoImage`` """
        assert self._display is not None
        return self._display

    @property
    def source(self) -> np.ndarray:
        """ The last BGR numpy frame stored under `set_source_image` """
        assert self._source is not None
        return self._source

    @property
    def scale(self) -> int:
        """ The current display scale as a whole-number percentage """
        return int(self._scale * 100)

    def set_source_image(self, name: str, image: np.ndarray) -> None:
        """ Store the composed preview frame under a name so it can later be rendered or saved

        Parameters
        ----------
        name
            Name under which to store the image; used later when rendering or saving
        image
            BGR numpy array of the preview frame to store
        """
        logger.debug("[_Image] Setting source image. name: '%s', shape: %s", name, image.shape)
        self._source = image

    def set_display_image(self) -> None:
        """ Rebuild the RGB ``PhotoImage`` from source, resizing when scaled """
        logger.debug("[_Image] Setting display image. Scale: %s", self._scale)
        image = self.source[..., 2::-1]  # TO RGB
        if self._scale not in (0.0, 1.0):  # Scale will be 0,0 on initial load in GUI
            interpolator = self._interpolation if self._scale > 1.0 else cv2.INTER_NEAREST
            dims = (int(round(self.source.shape[1] * self._scale, 0)),
                    int(round(self.source.shape[0] * self._scale, 0)))
            image = cv2.resize(image, dims, interpolation=interpolator)
        self._display = ImageTk.PhotoImage(Image.fromarray(image))
        logger.debug("[_Image] Set display image. Size: %s",
                     (self._display.width(), self._display.height()))

    def set_scale(self, scale: float) -> bool:
        """ Update the scale (zoom) factor

        Parameters
        ----------
        scale
            New scale factor to apply; If it is equal to the current value, no action is taken

        Returns
        -------
        ``True`` only if the value actually changed, otherwise ``False``
        """
        if self._scale == scale:
            return False
        logger.debug("[_Image] Setting scale: %s", scale)
        self._scale = scale
        return True

    def set_interpolation(self, interpolation: int) -> bool:
        """ Set the OpenCV resampling filter

        Parameters
        ----------
        interpolation
            New OpenCV resampling filter ENUM to use; If it is equal to the current value, no
            action is taken

        Returns
        -------
        ``True`` only if it actually changed, otherwise ``False``

        """
        if self._interpolation == interpolation:
            return False
        logger.debug("[_Image] Setting interpolation: %s", interpolation)
        self._interpolation = interpolation
        return True

    def save_preview(self, *args) -> None:
        """ Write the current source frame as a PNG standalone saves to argv's directory, embedded
        writes to the caller path

        Parameters
        ----------
        args
            Tuple containing either the key press event (Ctrl+s shortcut), the tk variable
            arguments (standalone save button press) or the folder location (GUI save button
            press). If the first argument is a tk.Event object, it's ignored and the
            caller path is used instead
        """
        if self._is_standalone and not self._save_var.get() and not isinstance(args[0], tk.Event):
            return
        if self._is_standalone:
            root_path = os.path.join(os.path.realpath(os.path.dirname(sys.argv[0])))
        else:
            root_path = T.cast(str, args[0])
        now = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        filename = os.path.join(root_path, f"preview_{now}.png")
        cv2.imwrite(filename, self.source)
        print("\x1b[2K", end="\r")  # Clear last line
        logger.info("Saved preview to: '%s'", filename)
        if self._is_standalone:
            self._save_var.set(False)


class _Bindings():  # pylint:disable=too-few-public-methods
    """ Wire the canvas and root window for wheel/scroll zoom, click-drag pan and keyboard move/
    save/interpolation actions

    This is a plain helper object assembled by `PreviewTk` from the three collaborators it needs
    (canvas, taskbar and image). It binds mouse-wheel or button-4/5 zoom on Linux versus MouseWheel
    elsewhere, plus click-to-pan dragging, arrow-key movement and the key-driven trigger bindings

    Parameters
    ----------
    canvas
        The ``_PreviewCanvas`` whose views are moved when panning or scrolling
    taskbar
        The ``_Taskbar`` holding `interpolator_var` so the ``i`` key can cycle resampling modes
    image
        The ``_Image`` owning `save_preview` for the Ctrl+s shortcut
    is_standalone
        Whether to bind keys at all; embedded panels skip per-key bindings to avoid stealing focus
        in another GUI

    Notes
    -----
    Key bindings only apply in standalone mode; mouse zoom differs on Linux (button-4/5) versus
    other platforms (MouseWheel)
    """
    def __init__(self,
                 canvas: _PreviewCanvas,
                 taskbar: _Taskbar,
                 image: _Image,
                 is_standalone: bool) -> None:
        logger.debug(parse_class_init(locals()))
        self._canvas = canvas
        self._taskbar = taskbar
        self._image = image
        self._is_standalone = is_standalone
        self._drag_data: list[float] = [0., 0.]
        self._set_mouse_bindings()
        self._set_key_bindings()

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        return (f"{self.__class__.__name__}("
                f"canvas={self._canvas!r}, "
                f"taskbar={self._taskbar!r}, "
                f"image={self._image!r}, "
                f"is_standalone={self._is_standalone!r})")

    def _on_bound_zoom(self, event: tk.Event) -> None:
        """ Clamp and set scale from a wheel/scroll/+/- press so the image zooms toward the cursor

        Parameters
        ----------
        event
            Event that triggered the zoom action.
        """
        if event.keysym in ("KP_Add", "plus") or event.num == 4 or event.delta > 0:
            scale = min(self._taskbar.max_scale, self._image.scale + 25)
        else:
            scale = max(self._taskbar.min_scale, self._image.scale - 25)
        logger.trace(  # type:ignore[attr-defined]
            "[_Bindings] Bound zoom action: (event: %s, scale: %s)", event, scale
            )
        self._taskbar.scale_var.set(f"{scale}%")

    def _on_mouse_click(self, event: tk.Event) -> None:
        """ Record the click position as normalized coordinates to anchor the subsequent drag pan

        Parameters
        ----------
        event
            Event that triggered the click action
        """
        self._drag_data = [event.x / self._image.display_image.width(),
                           event.y / self._image.display_image.height()]
        logger.trace(  # type:ignore[attr-defined]
            "[_Bindings] Mouse click action: (event: %s, drag_data: %s)",
            event, self._drag_data
            )

    def _on_mouse_drag(self, event: tk.Event) -> None:
        """ Pan the image by the delta between the current and last recorded mouse positions

        Parameters
        ----------
        event
            Event that triggered the drag action
        """
        location_x = event.x / self._image.display_image.width()
        location_y = event.y / self._image.display_image.height()
        if self._canvas.xview() != (0.0, 1.0):
            to_x = min(1.0, max(0.0, self._drag_data[0] - location_x + self._canvas.xview()[0]))
            self._canvas.xview_moveto(to_x)
        if self._canvas.yview() != (0.0, 1.0):
            to_y = min(1.0, max(0.0, self._drag_data[1] - location_y + self._canvas.yview()[0]))
            self._canvas.yview_moveto(to_y)
        self._drag_data = [location_x, location_y]

    def _on_key_move(self, event: tk.Event) -> None:
        """ Nudge the view along the x/y axis by a quarter of the visible fraction per arrow press

        Parameters
        ----------
        event
            Event that triggered the key press action
        """
        move_axis = self._canvas.xview if event.keysym in ("Left", "Right") else self._canvas.yview
        visible = move_axis()[1] - move_axis()[0]
        amount = -visible / 25 if event.keysym in ("Up", "Left") else visible / 25
        logger.trace(  # type:ignore[attr-defined]
            "[_Bindings] Key move event: (event: %s, move_axis: %s, visible: %s, amount: %s)",
            move_axis, visible, amount)
        move_axis(tk.MOVETO, min(1.0, max(0.0, move_axis()[0] + amount)))

    def _set_mouse_bindings(self) -> None:
        """ Bind wheel/scroll to zoom and click-drag to pan on the canvas image """
        logger.debug("[_Bindings] Binding mouse events")
        if system() == "Linux":
            self._canvas.tag_bind(self._canvas.image_id, "<Button-4>", self._on_bound_zoom)
            self._canvas.tag_bind(self._canvas.image_id, "<Button-5>", self._on_bound_zoom)
        else:
            self._canvas.bind("<MouseWheel>", self._on_bound_zoom)
        self._canvas.tag_bind(self._canvas.image_id, "<Button-1>", self._on_mouse_click)
        self._canvas.tag_bind(self._canvas.image_id, "<B1-Motion>", self._on_mouse_drag)
        logger.debug("[_Bindings] Bound mouse events")

    def _set_key_bindings(self) -> None:
        """ Bind arrows to move, +/- to zoom, Ctrl+s to save and i to cycle interpolators """
        if not self._is_standalone:
            # Don't bind keys for GUI as it adds complication
            return
        logger.debug("[_Bindings] Binding key events")
        root = self._canvas.winfo_toplevel()
        for key in ("Left", "Right", "Up", "Down"):
            root.bind(f"<{key}>", self._on_key_move)
        for key in ("Key-plus", "Key-minus", "Key-KP_Add", "Key-KP_Subtract"):
            root.bind(f"<{key}>", self._on_bound_zoom)
        root.bind("<Control-s>", self._image.save_preview)
        root.bind("<i>", self._taskbar.cycle_interpolators)
        logger.debug("[_Bindings] Bound key events")


class PreviewTk(PreviewBase):
    """ Interactive preview renderer that draws training frames into a window as either a
    standalone viewer or an embedded panel

    This is the rich Tkinter implementation of ``PreviewBase``: it builds a taskbar, scrollable
    canvas and image helper, wires up mouse/keyboard bindings, then runs the inherited `_launch`
    loop that waits for new previews from `PreviewBuffer` and renders them on demand. It can run
    alone as a fullscreen window (its real size is captured while iconified) or be embedded inside
    another GUI when given a parent frame

    Parameters
    ----------
    preview_buffer
        The thread-safe store of named previews being rendered; new frames are read through its
        `get_images`
    parent, optional
        A ``tk.Widget`` to embed in so the viewer becomes a non-fullscreen panel inside another
        GUI; when omitted a standalone window is created. Default: ``None``
    taskbar, optional
        A caller-supplied option-bar frame used only for embedded panels; ``None`` lets `PreviewTk`
        build its own standalone bar. Default: ``None``
    triggers, optional
        Optional mapping of trigger name to threading event (for example refresh or save); when
        omitted key-driven triggers are unavailable and only the private shutdown hook is
        honored. Default: ``None``

    Notes
    -----
    After setup it calls `_launch`, which waits for new previews from the buffer and renders each
    one, restoring a standalone window to its measured size on first display
    """
    def __init__(self,
                 preview_buffer: PreviewBuffer,
                 parent: tk.Widget | None = None,
                 taskbar: ttk.Frame | None = None,
                 triggers: TriggerType | None = None) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(preview_buffer, triggers=triggers)
        self._is_standalone = parent is None
        self._initialized = False
        self._root = parent if parent is not None else tk.Tk()
        self._master_frame = tk.Frame(self._root)
        self._taskbar = _Taskbar(self._master_frame, taskbar)
        self._screen_dimensions = self._get_geometry()
        self._canvas = _PreviewCanvas(self._master_frame,
                                      self._taskbar.scale_var,
                                      self._screen_dimensions,
                                      self._is_standalone)
        self._image = _Image(self._taskbar.save_var, self._is_standalone)
        _Bindings(self._canvas, self._taskbar, self._image, self._is_standalone)
        self._taskbar.scale_var.trace("w", self._set_scale)
        self._taskbar.interpolator_var.trace("w", self._set_interpolation)

        self._process_triggers()
        if self._is_standalone:
            self.pack(fill=tk.BOTH, expand=True)
        self._output_helptext()
        self._launch()

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        retval = super().__repr__()[:-1]
        return (f"{retval}, "
                f"parent={self._root}, "
                f"taskbar={self._taskbar!r})")

    @property
    def master_frame(self) -> tk.Frame:
        """ The frame that holds the taskbar, canvas and image helper as a unit """
        return self._master_frame

    def pack(self, *args, **kwargs) -> None:
        """ Re-pack the whole preview into a caller-supplied position so an embedded panel can be
        laid out freely """
        logger.debug("[PreviewTk] Packing master frame: (args: %s, kwargs: %s)", args, kwargs)
        self._master_frame.pack(*args, **kwargs)

    def save(self, location: str) -> None:
        """ Save the current preview image to a PNG at the given path

        Parameters
        ----------
        location
            The path where the preview image should be saved
        """
        self._image.save_preview(location)

    def remove_option_controls(self) -> None:
        """ Destroy and drop the option-bar widgets when embedding in another GUI """
        self._taskbar.destroy_widgets()

    def _output_helptext(self) -> None:
        """ Log a short key-bindings banner for standalone viewers instructing end users """
        if not self._is_standalone:
            return
        logger.info("---------------------------------------------------")
        logger.info("  Preview key bindings:")
        logger.info("    Zoom:              +/-")
        logger.info("    Toggle Zoom Mode:  i")
        logger.info("    Move:              arrow keys")
        logger.info("    Save Preview:      Ctrl+s")
        logger.info("---------------------------------------------------")

    def _get_geometry(self) -> tuple[int, int]:
        """ Capture the available screen dimensions used as the size ceiling for standalone window

        Returns
        --------
        width
            The width of the screen in pixels
        height
            The height of the screen in pixels
        """
        if not self._is_standalone:
            root = self._root.winfo_toplevel()  # Get dims of whole GUI
            retval = root.winfo_width(), root.winfo_height()
            logger.debug("[PreviewTk] Obtained frame geometry: %s", retval)
            return retval
        assert isinstance(self._root, tk.Tk)
        logger.debug("[PreviewTk] Obtaining screen geometry")
        self._root.update_idletasks()
        self._root.attributes("-fullscreen", True)
        self._root.state("iconic")
        retval = self._root.winfo_width(), self._root.winfo_height()
        self._root.attributes("-fullscreen", False)
        self._root.state("withdraw")
        logger.debug("[PreviewTk] Obtained screen geometry: %s", retval)
        return retval

    def _set_min_max_scales(self) -> None:
        """ Compute the slider's min/max scale limits from the image and screen size and apply """
        logger.debug("[PreviewTk] Calculating minimum scale for screen dimensions %s",
                     self._screen_dimensions)
        half_screen = tuple(x // 2 for x in self._screen_dimensions)
        min_scales = (half_screen[0] / self._image.source.shape[1],
                      half_screen[1] / self._image.source.shape[0])
        min_scale = min(1.0, *min_scales)
        min_scale = (ceil(min_scale * 10)) * 10

        eight_screen = tuple(x * 8 for x in self._screen_dimensions)
        max_scales = (eight_screen[0] / self._image.source.shape[1],
                      eight_screen[1] / self._image.source.shape[0])
        max_scale = min(8.0, max(1.0, min(max_scales)))
        max_scale = (floor(max_scale * 10)) * 10

        logger.debug("[PreviewTk] Calculated minimum scale: %s, maximum_scale: %s",
                     min_scale, max_scale)
        self._taskbar.set_min_max_scale(min_scale, max_scale)

    def _initialize_window(self) -> None:
        """ Restore the hidden probe-sized window to its measured real size """
        logger.debug("[PreviewTk] Initializing window")
        assert isinstance(self._root, tk.Tk)
        width = min(self._master_frame.winfo_reqwidth(), self._screen_dimensions[0])
        height = min(self._master_frame.winfo_reqheight(), self._screen_dimensions[1])
        self._set_min_max_scales()
        self._root.state("normal")
        self._root.geometry(f"{width}x{height}")
        self._root.protocol("WM_DELETE_WINDOW", lambda: None)  # Intercept close window
        self._initialized = True
        logger.debug("[PreviewTk] Initialized window: (width: %s, height: %s)", width, height)

    def _update_image(self, center_image: bool = False) -> None:
        """ Refresh the canvas with a new composed display image (recenters when newly initialized)

        Parameters
        ----------
        center_image
            If ``True`` recenter the image on the canvas, otherwise, leave it centered where it was
        """
        logger.debug("[PreviewTk] Updating image (center_image: %s)", center_image)
        self._image.set_display_image()
        self._canvas.set_image(self._image.display_image, center_image)
        logger.debug("[PreviewTk] Updated image")

    def _convert_fit_scale(self) -> str:
        """ Get the percentage scale that fits the whole image into the current canvas dimensions

        Returns
        -------
        str
            The percentage scale that fits the whole image into the current canvas dimensions
        """
        logger.debug("[PreviewTk] Converting 'Fit' scaling")
        width_scale = self._canvas.width / self._image.source.shape[1]
        height_scale = self._canvas.height / self._image.source.shape[0]
        scale = min(width_scale, height_scale) * 100
        retval = f"{floor(scale)}%"
        logger.debug("[PreviewTk] Converted 'Fit' scaling: (width_scale: %s, height_scale: %s, "
                     "scale: %s, retval: '%s'",
                     width_scale, height_scale, scale, retval)
        return retval

    def _set_scale(self, *args) -> None:  # pylint:disable=unused-argument
        """ Trace callback for the scale variable; applies a new scale and re-renders on change """
        txt_scale = self._taskbar.scale_var.get()
        logger.debug("[PreviewTk] Setting scale: '%s'", txt_scale)
        txt_scale = self._convert_fit_scale() if txt_scale == "Fit" else txt_scale
        scale = int(txt_scale[:-1])  # Strip percentage and convert to int
        logger.debug("[PreviewTk] Got scale: %s", scale)
        if self._image.set_scale(scale / 100):
            logger.debug("[PreviewTk] Updating for new scale")
            self._taskbar.slider_var.set(scale)
            self._update_image(center_image=True)

    def _set_interpolation(self, *args) -> None:  # pylint:disable=unused-argument
        """ Trace callback for the interpolation variable; swaps resampling filter on change """
        interpolator = self._taskbar.interpolator_var.get()
        if not self._image.set_interpolation(interpolator) or self._image.scale <= 1.0:
            return
        self._update_image(center_image=False)

    def _process_triggers(self) -> None:
        """ Bind each keymap entry to its physical key on the root window to drive the preview """
        if self._triggers is None:  # Don't need triggers for GUI
            return
        logger.debug("[PreviewTk] Processing triggers")
        root = self._canvas.winfo_toplevel()
        for key in self._keymaps:
            bind_key = "Return" if key == "enter" else key
            logger.debug("[PreviewTk] Adding trigger for key: '%s'", bind_key)

            root.bind(f"<{bind_key}>", self._on_keypress)
        logger.debug("[PreviewTk] Processed triggers")

    def _on_keypress(self, event: tk.Event) -> None:
        """ Dispatch a pressed key (r refreshes; others map through shared keymap)

        Parameters
        ----------
        event
            The key press event data
        """
        if self._triggers is None:  # Don't need triggers for GUI
            return
        keypress = "enter" if event.keysym == "Return" else event.keysym
        key = T.cast("TriggerKeysType", keypress)
        logger.debug("[PreviewTk] Processing keypress '%s'", key)
        if key == "r":
            print("\x1b[2K", end="\r")  # Clear last line
            logger.info("Refresh preview requested...")

        self._triggers[self._keymaps[key]].set()
        logger.debug("[PreviewTk] Processed keypress '%s'. Set event for '%s'",
                     key, self._keymaps[key])

    def _display_preview(self) -> None:
        """ Pull images from buffer and schedule on a timer until shutdown or first display """
        if self._should_shutdown:
            self._root.destroy()

        if not self._buffer.is_updated:
            self._root.after(1000, self._display_preview)
            return

        for name, image in self._buffer.get_images():
            logger.debug("[PreviewTk] Updating image: (name: '%s', shape: %s)", name, image.shape)
            if self._is_standalone and not self._title:
                assert isinstance(self._root, tk.Tk)
                self._title = name
                logger.debug("[PreviewTk] Setting title: '%s;", self._title)
                self._root.title(self._title)
            self._image.set_source_image(name, image)
            self._update_image(center_image=not self._initialized)

        self._root.after(1000, self._display_preview)

        if not self._initialized and self._is_standalone:
            self._initialize_window()
            self._root.mainloop()
        if not self._initialized:  # Set initialized to True for GUI
            self._set_min_max_scales()
            self._taskbar.scale_var.set("Fit")
            self._initialized = True


__all__ = get_module_objects(__name__)
