#!/usr/bin/env python3
""" Handles the downloading of model weights files from remote resources and caching locally """
from __future__ import annotations

import logging
import os
import sys
import typing as T
import zipfile
from socket import timeout as socket_timeout, error as socket_error
from urllib import request, error as urlliberror

from tqdm import tqdm
from lib.logger import parse_class_init
from lib.utils import get_module_objects, PROJECT_ROOT

if T.TYPE_CHECKING:
    from http.client import HTTPResponse

logger = logging.getLogger(__name__)


class GetWeights():
    """Check for models in the cache path. If available, return the path, if not available,
    get, unzip and install weights file

    Parameters
    ----------
    model_filename
        The name of the model to be loaded (see notes below)
    version
        The version ID of the weights to load. Default: 0
    git_model_id, optional
        The second digit in the github tag that identifies this model. See
        https://github.com/deepfakes-models/faceswap-models for more information. Default:
        ``None`` (load from huggingface)

    Notes
    ------
    Models must have a certain naming convention: `<model_name>_v<version_number>.<extension>`
    (eg: `s3fd_v1.pb`). The version number should not be passed as part of the model filename, it
    will be automatically added to the model file by the given version ID. For example for the 3rd
    version of s3fd.pth you would request:
    GetWeights("s3fd_keras.pth", 3, 11)
    This will return the file s3fd_keras_v3.pth

    Multiple models can exist within the model_filename. They should be passed as a list and follow
    the same naming convention as above. Any differences in filename should occur BEFORE the
    version number: `<model_name><differentiating_information>_v<version_number>.<extension>` (eg:
    `["mtcnn_det.1_v1.py", "mtcnn_det.2_v1.py", "mtcnn_det.3_v1.py"]`, `["resnet_ssd_v1.caffemodel"
    ,"resnet_ssd_v1.prototext"]`

    Example
    -------
    >>> from lib.utils import GetWeights
    >>> model_downloader = GetWeights("s3fd_keras.h5", 2)
    """

    def __init__(self, model_filename: str | list[str],
                 version: int = 0,
                 git_model_id: int | None = None) -> None:
        logger.debug(parse_class_init(locals()))
        self._filenames = self._get_model_filenames(model_filename, version)
        self._version = version
        self._cache_dir = os.path.join(PROJECT_ROOT, ".fs_cache")
        self._get(git_model_id)

    @property
    def _model_identifier(self) -> str:
        """The full model name from the filename(s). This is any common prefix (for zips with
        multiple files) or the filename, with the extension removed """
        common_prefix = os.path.commonprefix(self._filenames).rstrip("_")
        retval = os.path.splitext(common_prefix)[0]
        logger.trace("[GetWeights] full name: %s", repr(retval))  # type:ignore[attr-defined]
        return retval

    @property
    def model_path(self) -> str | list[str]:
        """The model path(s) in the cache folder.

        Example
        -------
        >>> from lib.utils import GetWeights
        >>> model_downloader = GetWeights("s3fd_keras.pth", 2)
        >>> model_downloader.model_path
        '/path/to/s3fd_keras_v2.pth'
        """
        paths = [os.path.join(self._cache_dir, fname) for fname in self._filenames]
        retval: str | list[str] = paths[0] if len(paths) == 1 else paths
        logger.trace("[GetWeights] path: %s", repr(retval))  # type:ignore[attr-defined]
        return retval

    @property
    def _model_exists(self) -> bool:
        """``True`` if the model exists in the cache folder otherwise ``False``."""
        if isinstance(self.model_path, list):
            retval = all(os.path.exists(pth) for pth in self.model_path)
        else:
            retval = os.path.exists(self.model_path)
        logger.trace("[GetWeights] exists: %s", repr(retval))  # type:ignore[attr-defined]
        return retval

    @classmethod
    def _get_model_filenames(cls, filenames: list[str] | str, version: int) -> list[str]:
        """ constructs the full model filename(s) by appending the version number to each input
        filename if version is greater than 0

        Parameters
        ----------
        filenames
            The input model filename(s). Can be a single string or a list of strings.
        version
            The version number to append to each filename

        Returns
        -------
        A list of fully constructed filenames with the version number appended.
        """
        retval = filenames if isinstance(filenames, list) else [filenames]
        if version:
            split = [os.path.splitext(x) for x in retval]
            retval = [f"{fname}_v{version}{ext}" for fname, ext in split]
        logger.debug("[GetWeights] Model '%s' filename from version %s: %s",
                     filenames, version, retval)
        return retval

    def _get(self, git_model_id: int | None) -> None:
        """Check the model exists, if not, download the model, unzip it and place it in the
        model's cache folder.

        Parameters
        ----------
        git_model_id
            The second digit in the github tag that identifies this model if the model is stored
            within the deepdakes-models github repository
        """
        if self._model_exists:
            logger.debug("[GetWeights] Model exists: %s", repr(self.model_path))
            return

        if git_model_id is None:
            weights_from_huggingface(self._cache_dir, self._model_identifier)
        else:
            weights_from_github(self._cache_dir, self._model_identifier, git_model_id)


class Downloader:
    """ Downloads a zipped model file from the given resource and de-compresses it to faceswap's
    cache

    Parameters
    ----------
    destination_folder
        The directory to cache the model in
    url
        The URL to download the zip file from. The filename must be derivable from the end of the
        URL
    display_url
        The Repo URL to display for logging. Default "" (empty string: use download URL)
    retries
        Number of times to retry downloading before failing. Default: 6
    chunk_size
        Chunk size for downloading and unzipping. Default: 1024
    """
    def __init__(self,
                 destination_folder: str,
                 url: str,
                 display_url: str = "",
                 retries: int = 6,
                 chunk_size: int = 1024) -> None:
        logger.debug(parse_class_init(locals()))
        self._cache_dir = destination_folder
        assert os.path.splitext(url)[-1] == ".zip"
        self._url = url
        self._display_url = display_url if display_url else url
        self._model_name = os.path.basename(url)
        self._model_zip_path = os.path.join(self._cache_dir, os.path.basename(url))

        self._retries = retries
        self._chunk_size = chunk_size

    @property
    def _downloaded_bytes(self) -> int:
        """ How many bytes have already been downloaded. """
        zip_file = self._model_zip_path
        retval = os.path.getsize(zip_file) if os.path.exists(zip_file) else 0
        logger.trace("[GetWeights] Partial size: %s", retval)  # type:ignore[attr-defined]
        return retval

    def _validate_zip(self, expected_length: int) -> None:
        """ Validate that the downloaded zip is the correct size and is not corrupt

        Parameters
        ----------
        expected_length
            The number of bytes that the zip file should be

        Raises
        ------
        RuntimeError if zip file failed validation
        """
        final_size = self._downloaded_bytes
        logger.info("Validating: '%s'...", os.path.basename(self._model_zip_path))
        logger.debug("[GetWeights] Downloaded '%s'. Expected: %s, got: %s",
                     os.path.basename(self._model_zip_path), expected_length, final_size)

        if final_size < expected_length:
            raise RuntimeError("Truncated download. Resuming")

        is_error = False
        if final_size > expected_length:
            is_error = True
        elif not zipfile.is_zipfile(self._model_zip_path):
            is_error = True
        else:
            try:
                with zipfile.ZipFile(self._model_zip_path) as zf:
                    is_error = zf.testzip() is not None
            except zipfile.BadZipFile:
                is_error = True

        if is_error and os.path.exists(self._model_zip_path):
            os.remove(self._model_zip_path)

        if is_error:
            raise RuntimeError("Corrupted download. Retrying")

    def _write_zipfile(self, response: HTTPResponse, downloaded_size: int) -> None:
        """ Write the model zip file to disk.

        Parameters
        ----------
        response
            The response from the model download task
        downloaded_size
            The amount of bytes downloaded so far
        """
        content_length = response.getheader("content-length")
        content_length = "0" if content_length is None else content_length
        length = int(content_length) + downloaded_size
        if length == downloaded_size:
            self._validate_zip(length)
            logger.info("Zip already exists. Skipping download")
            return
        write_type = "wb" if downloaded_size == 0 else "ab"

        with open(self._model_zip_path, write_type) as out_file:
            p_bar = tqdm(desc="Downloading",
                         unit="B",
                         total=length,
                         unit_scale=True,
                         unit_divisor=1024,
                         leave=False)
            if downloaded_size != 0:
                p_bar.update(downloaded_size)
            while True:
                buffer = response.read(self._chunk_size)
                if not buffer:
                    break
                p_bar.update(len(buffer))
                out_file.write(buffer)
            p_bar.close()

        self._validate_zip(length)

    def _download_model(self) -> None:
        """ Download the model zip from github to the cache folder. """
        logger.info("Downloading: '%s' from: %s", self._model_name, self._display_url)
        for attempt in range(self._retries):
            try:
                downloaded_size = self._downloaded_bytes
                req = request.Request(self._url)
                if downloaded_size != 0:
                    req.add_header("Range", f"bytes={downloaded_size}-")
                with request.urlopen(req, timeout=10) as response:
                    logger.debug("[GetWeights] header info: {%s}", response.info())
                    logger.debug("[GetWeights] Return Code: %s", response.getcode())
                    self._write_zipfile(response, downloaded_size)
                break
            except (socket_error, socket_timeout,
                    urlliberror.HTTPError, urlliberror.URLError, RuntimeError) as err:
                if attempt + 1 < self._retries:
                    logger.warning("Error downloading model (%s). Retrying %s of %s...",
                                   str(err), attempt + 2, self._retries)
                else:
                    logger.error("Failed to download model. Exiting. (Error: '%s', URL: '%s')",
                                 str(err), self._url)
                    logger.info("You can try running again to resume the download.")
                    logger.info("Alternatively, you can manually download the model from: %s "
                                "and unzip the contents to: %s",
                                self._url, self._cache_dir)
                    sys.exit(1)

    def _write_model(self, zip_file: zipfile.ZipFile) -> None:
        """ Extract files from zip file and write, with progress bar.

        Parameters
        ----------
        zip_file
            The downloaded model zip file
        """
        length = sum(f.file_size for f in zip_file.infolist())
        f_names = zip_file.namelist()
        logger.debug("[GetWeights] Zipfile: Filenames: %s, Total Size: %s", f_names, length)
        assert tqdm is not None
        p_bar = tqdm(desc="Decompressing",
                     unit="B",
                     total=length,
                     unit_scale=True,
                     unit_divisor=1024,
                     leave=False)
        for fname in f_names:
            out_fname = os.path.join(self._cache_dir, fname)
            logger.debug("[GetWeights] Extracting from: '%s' to '%s'",
                         self._model_zip_path, out_fname)
            zipped = zip_file.open(fname)
            with open(out_fname, "wb") as out_file:
                while True:
                    buffer = zipped.read(self._chunk_size)
                    if not buffer:
                        break
                    p_bar.update(len(buffer))
                    out_file.write(buffer)
        p_bar.close()

    def _unzip_model(self) -> None:
        """ Unzip the model file to the cache folder """
        logger.info("Extracting: '%s'...", self._model_name)
        try:
            with zipfile.ZipFile(self._model_zip_path, "r") as zip_file:
                self._write_model(zip_file)
        except Exception as err:  # pylint:disable=broad-except
            logger.error("Unable to extract model file: %s", str(err))
            sys.exit(1)

    def download(self) -> None:
        """ Download the model, unzip it and place it in faceswap's cache folder. """
        self._download_model()
        self._unzip_model()
        os.remove(self._model_zip_path)


def weights_from_github(folder: str,
                        model_name: str,
                        git_model_id: int,
                        repo: str = "deepfakes-models/faceswap-models",
                        retries=6,
                        chunk_size=1024) -> None:
    """ Download a model from deepfakes-models releases and unzip to the cache path.

    Parameters
    ----------
    folder
        The directory to cache the model in
    model_name
        The name of the model without the file extension. For zips that contain multiple models
        this will be the common prefix. Otherwise it will be the filename.
    git_model_id
        The second digit in the github tag that identifies this model. See
        https://github.com/deepfakes-models/faceswap-models for more information
    repo
        The Github repository identifier. Default: "deepfakes-models/faceswap-models"
    retries
        Number of times to retry downloading before failing. Default: 6
    chunk_size
        Chunk size for downloading and unzipping. Default: 1024
    """
    url_base = f"https://github.com/{repo}/releases"
    version = int(model_name[model_name.rfind("_") + 2:])
    tag = f"v{git_model_id}.{version}"
    dsp_url = f"{url_base}/{tag}"
    url = f"{url_base}/download/{tag}/{model_name}.zip"
    Downloader(folder, url, display_url=dsp_url, retries=retries, chunk_size=chunk_size).download()


def weights_from_huggingface(folder: str,
                             model_name: str,
                             repo: str = "deepfakes/faceswap",
                             version: int = 0,
                             retries: int = 6,
                             chunk_size: int = 1024) -> None:
    """ Download a model from HuggingFace and unzip to the cache path.

    Parameters
    ----------
    folder
        The directory to cache the model in
    model_name
        The name of the model without the file extension. For zips that contain multiple models
        this will be the common prefix. Otherwise it will be the filename.
    repo
        The HuggingFace repository identifier. Default: "deepfakes/faceswap"
    version
        The version ID of the model. Default: 0
    retries
        Number of times to retry downloading before failing. Default: 6
    chunk_size
        Chunk size for downloading and unzipping. Default: 1024

    Example
    -------
    >>> from lib.utils import HuggingFaceDownloader
    >>> model_downloader = HuggingFaceDownloader(
    ...     folder=".fs_cache",
    ...     model_name="resnet50",
    ...     repo="deepfakes/faceswap",
    ...     version=0,
    ... )
    """
    url_base = "https://huggingface.co"
    model_file = model_name if version == 0 else f"{model_name}_v{version}"
    dsp_url = f"{url_base}/buckets/{repo}"
    url = f"{dsp_url}/resolve/{model_file}.zip"
    Downloader(folder, url, display_url=dsp_url, retries=retries, chunk_size=chunk_size).download()


__all__ = get_module_objects(__name__)
