#!/usr/bin/env python3
"""Plugin loader for Faceswap extract, training and convert tasks"""
from __future__ import annotations
import ast
import logging
import os
import typing as T

from importlib import import_module

from lib.utils import full_path_split, get_module_objects, PROJECT_ROOT

if T.TYPE_CHECKING:
    from collections.abc import Callable
    from plugins.extract.base import ExtractPlugin
    from plugins.train.model.base import ModelPlugin
    from plugins.train.trainer.base import TrainerPlugin

logger = logging.getLogger(__name__)


def _plugins_from_files(file_list: list[str], plugin_classes: list[str]):
    """Parse the given file list for instances of the given plugin classes

    Parameters
    ----------
    file_list
        list of full paths to python files to scan for instances of the given plugin classes

    Returns
    -------
    list of relative import paths for the discovered plugins from the project root
    """
    retval: list[str] = []
    for f_path in file_list:
        try:
            with open(f_path, "r", encoding="utf-8") as pfile:
                tree = ast.parse(pfile.read())
        except Exception:  # pylint:disable=broad-except
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                if not isinstance(base, ast.Name):
                    continue
                if base.id in plugin_classes:
                    rel_path = os.path.splitext(f_path.replace(PROJECT_ROOT, "")[1:])[0]
                    retval.append(".".join(full_path_split(rel_path) + [node.name]))
    return retval


def _get_extractors() -> dict[str, list[str]]:
    """Obtain a dictionary of all available extraction plugins by plugin type

    Returns
    -------
    A list of all available plugins for each extraction plugin type
    """
    root = os.path.join(PROJECT_ROOT, "plugins", "extract")
    folders = sorted(os.path.join(root, f) for f in os.listdir(root)
                     if os.path.isdir(os.path.join(root, f))
                     and not f.startswith("_"))
    retval: dict[str, list[str]] = {}
    for fld in folders:
        files = sorted(f for fname in os.listdir(fld)
                       if os.path.isfile(f := os.path.join(fld, fname))
                       and fname.endswith(".py")
                       and not fname.startswith("_")
                       and not fname.endswith("_defaults.py"))
        mods = _plugins_from_files(files, ["ExtractPlugin", "FacePlugin"])
        if mods:
            retval[os.path.basename(fld)] = list(sorted(mods))
    return retval


def _get_models() -> list[str]:
    """Obtain a list of all trainable model plugins

    Returns
    -------
    All available faceswap model names
    """
    root = os.path.join(PROJECT_ROOT, "plugins", "train", "model")
    files = sorted(p for f in os.listdir(root)
                   if os.path.isfile(p := os.path.join(root, f))
                   and f.endswith(".py")
                   and not f.startswith("_")
                   and not f.endswith("_defaults.py"))

    return _plugins_from_files(files, ["ModelPlugin"])


class PluginLoader():
    """Retrieve, or get information on, Faceswap plugins

    Return a specific plugin, list available plugins, or get the default plugin for a
    task.

    Example
    -------
    >>> from plugins.plugin_loader import PluginLoader
    >>> align_plugins = PluginLoader.get_available_extractors('align')
    >>> aligner = PluginLoader.get_aligner('cv2-dnn')
    """
    extract_plugins = _get_extractors()
    model_plugins = _get_models()

    @classmethod
    def _import_plugin(cls, import_path: str) -> T.Any:
        """Import the plugin class from the given full plugin import path

        Parameters
        ----------
        path
            The dot separated relative path to the plugin class

        Returns
        -------
        the imported plugin class
        """
        mod, obj = import_path.rsplit(".", maxsplit=1)
        logger.debug("[PluginLoader] Loading '%s' from '%s'", obj, mod)
        module = import_module(mod)
        retval = getattr(module, obj)
        logger.debug("[PluginLoader] Loaded plugin: %s", retval)
        return retval

    @classmethod
    def get_extractor(cls,
                      plugin_type: T.Literal["align", "detect", "identity", "mask"],
                      name: str) -> ExtractPlugin:
        """Return requested extractor plugin

        Parameters
        ----------
        type
            The type of extractor plugin to obtain
        name
            The name of the requested extractor plugin

        Returns
        -------
        An extraction plugin

        Raises
        ------
        ValueError
            If an invalid plugin type or plugin name is selected
        """
        if plugin_type not in cls.extract_plugins:
            raise ValueError(f"{plugin_type} is not a valid plugin type. Select from "
                             f"{list(cls.extract_plugins)}")
        plugins = cls.extract_plugins[plugin_type]
        mods = [p.split(".")[-2] for p in plugins]
        real_name = name.lower().replace("-", "_")
        if real_name not in mods:
            raise ValueError(f"{name} is not a valid {plugin_type} plugin. Select from {mods}")

        retval = cls._import_plugin(plugins[mods.index(real_name)])()
        logger.info("Loading %s from %s", plugin_type.title(), retval.name)
        return retval

    @classmethod
    def get_available_extractors(cls,
                                 extractor_type: T.Literal["align", "detect", "identity", "mask"],
                                 add_none: bool = False,
                                 extend_plugin: bool = False) -> list[str]:
        """Return a list of available extractors of the given type

        Parameters
        ----------
        extractor_type
            The type of extractor to return the plugins for
        add_none
            Append "none" to the list of returned plugins. Default: False
        extend_plugin
            Some plugins have configuration options that mean that multiple 'pseudo-plugins'
            can be generated based on their settings. An example of this is the bisenet-fp mask
            which, whilst selected as 'bisenet-fp' can be stored as 'bisenet-fp-face' and
            'bisenet-fp-head' depending on whether hair has been included in the mask or not.
            ``True`` will generate each pseudo-plugin, ``False`` will generate the original
            plugin name. Default: ``False``

        Returns
        -------
        A list of the available extractor plugin names for the given type
        """
        if extractor_type not in cls.extract_plugins:
            raise ValueError(f"{extractor_type} is not a valid plugin type. Select from "
                             f"{list(cls.extract_plugins)}")
        plugins = [x.split(".")[-2].replace("_", "-") for x in cls.extract_plugins[extractor_type]]
        if extend_plugin and extractor_type == "mask":
            extendable = ["bisenet-fp", "custom"]
            for plugin in extendable:
                if plugin not in plugins:
                    continue
                plugins.remove(plugin)
                plugins.extend([f"{plugin}_face", f"{plugin}_head"])
        plugins = sorted(plugins)
        if add_none:
            plugins.insert(0, "none")
        return plugins

    @classmethod
    def get_model_path(cls, name: str, module: bool = False) -> str:
        """ Obtain the full dot separated path to a training model plugin relative to the project
        root

        Parameters
        ----------
        name
            The name of the requested training model plugin
        module
            ``True`` to return the path to the containing module. ``False`` to return the path to
            the plugin object. Default: ``False``

        Returns
        -------
        The dot separated path to the training model relative to the project root
        """
        name = name.lower().replace("-", "_")
        mods = [p.split(".")[-2] for p in cls.model_plugins]
        if name not in mods:
            raise ValueError(f"{name} is not a valid train plugin. Select from "
                             f"{[x.replace('_', '-') for x in mods]}")

        retval = cls.model_plugins[mods.index(name)]
        if module:
            retval = retval.rsplit(".", maxsplit=1)[0]
        logger.debug("[PluginLoader] name: '%s', module: %s, path: %s", name, module, retval)
        return retval

    @classmethod
    def get_model(cls, name: str) -> type[ModelPlugin]:
        """Return requested training model plugin

        Parameters
        ----------
        name
            The name of the requested training model plugin

        Returns
        -------
        A training model plugin
        """
        path = cls.get_model_path(name, module=False)
        retval = cls._import_plugin(path)
        logger.info("Loading Model from %s plugin", path.rsplit(".", maxsplit=1)[-1])
        return retval

    @classmethod
    def get_available_models(cls) -> list[str]:
        """Return a list of available training models

        Returns
        -------
        A list of the available training model plugin names
        """
        return list(sorted(x.split(".")[-2].replace("_", "-") for x in cls.model_plugins))

    @staticmethod
    def get_default_model() -> str:
        """Return the default training model plugin name

        Returns
        -------
        The default faceswap training model
        """
        models = PluginLoader.get_available_models()
        return 'original' if 'original' in models else models[0]

    @staticmethod
    def _import(attr: str, name: str, disable_logging: bool):
        """Import the plugin's module

        Parameters
        ----------
        name
            The name of the requested plugin
        disable_logging
            Whether to disable the INFO log message that the plugin is being imported.

        Returns
        -------
        A plugin
        """
        name = name.replace("-", "_")
        ttl = attr.split(".")[-1].title()
        if not disable_logging:
            logger.info("Loading %s from %s plugin...", ttl, name.title())
        attr = "model" if attr == "Trainer" else attr.lower()
        mod = ".".join(("plugins", attr, name))
        module = import_module(mod)
        return getattr(module, ttl)

    @staticmethod
    def get_trainer(name: str, disable_logging: bool = False) -> type[TrainerPlugin]:
        """Return requested training trainer plugin

        Parameters
        ----------
        name
            The name of the requested training trainer plugin
        disable_logging
            Whether to disable the INFO log message that the plugin is being imported.
            Default: `False`

        Returns
        -------
        A training trainer plugin
        """
        return PluginLoader._import("train.trainer", name, disable_logging)

    @staticmethod
    def get_converter(category: str, name: str, disable_logging: bool = False) -> Callable:
        """Return requested converter plugin

        Converters work slightly differently to other faceswap plugins. They are created to do a
        specific task (e.g. color adjustment, mask blending etc.), so multiple plugins will be
        loaded in the convert phase, rather than just one plugin for the other phases.

        Parameters
        ----------
        name
            The name of the requested converter plugin
        disable_logging
            Whether to disable the INFO log message that the plugin is being imported.
            Default: `False`

        Returns
        -------
        A converter sub plugin
        """
        return PluginLoader._import(f"convert.{category}", name, disable_logging)

    @staticmethod
    def get_available_convert_plugins(convert_category: str, add_none: bool = True) -> list[str]:
        """Return a list of available converter plugins in the given category

        Parameters
        ----------
        convert_category
            The category of converter plugin to return the plugins for
        add_none
            Append "none" to the list of returned plugins. Default: True

        Returns
        -------
        A list of the available converter plugin names in the given category
        """

        convert_path = os.path.join(os.path.dirname(__file__),
                                    "convert",
                                    convert_category)
        converters = sorted(item.name.replace(".py", "").replace("_", "-")
                            for item in os.scandir(convert_path)
                            if not item.name.startswith("_")
                            and not item.name.endswith("defaults.py")
                            and item.name.endswith(".py"))
        if add_none:
            converters.insert(0, "none")
        return converters


__all__ = get_module_objects(__name__)
