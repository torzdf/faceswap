#!/usr/bin/env python3
""" Handles the downloading of model weights files from remote resources and caching locally """
from __future__ import annotations

import abc
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
    git_model_id
        The second digit in the github tag that identifies this model. See
        https://github.com/deepfakes-models/faceswap-models for more information

    Notes
    ------
    Models must have a certain naming convention: `<model_name>_v<version_number>.<extension>`
    (eg: `s3fd_v1.pb`).

    Multiple models can exist within the model_filename. They should be passed as a list and follow
    the same naming convention as above. Any differences in filename should occur AFTER the version
    number: `<model_name>_v<version_number><differentiating_information>.<extension>` (eg:
    `["mtcnn_det_v1.1.py", "mtcnn_det_v1.2.py", "mtcnn_det_v1.3.py"]`, `["resnet_ssd_v1.caffemodel"
    ,"resnet_ssd_v1.prototext"]`

    Example
    -------
    >>> from lib.utils import GetWeights
    >>> model_downloader = GetWeights("s3fd_keras_v2.h5", 11)
    """

    def __init__(self, model_filename: str | list[str], git_model_id: int) -> None:
        logger.debug(parse_class_init(locals()))
        if not isinstance(model_filename, list):
            model_filename = [model_filename]
        self._model_filename = model_filename
        self._cache_dir = os.path.join(PROJECT_ROOT, ".fs_cache")
        self._get(git_model_id)

    @property
    def _model_full_name(self) -> str:
        """The full model name from the filename(s)."""
        common_prefix = os.path.commonprefix(self._model_filename)
        retval = os.path.splitext(common_prefix)[0]
        logger.trace("[GetWeights] full name: %s", repr(retval))  # type:ignore[attr-defined]
        return retval

    @property
    def model_path(self) -> str | list[str]:
        """The model path(s) in the cache folder.

        Example
        -------
        >>> from lib.utils import GetWeights
        >>> model_downloader = GetWeights("s3fd_keras_v2.h5", 11)
        >>> model_downloader.model_path
        '/path/to/s3fd_keras_v2.h5'
        """
        paths = [os.path.join(self._cache_dir, fname) for fname in self._model_filename]
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

        if git_model_id is not None:
            GithubDownloader(self._cache_dir,
                             self._model_full_name,
                             git_model_id).download()
        else:
            raise ValueError("No model ID provided for download.")


class Downloader(abc.ABC):
    """ Downloads a zipped model file from the given resource and de-compresses it to faceswap's
    cache

    Parameters
    ----------
    folder
        The directory to cache the model in
    model_name
        The full path to the extracted location
    url
        The URL to download the zip file from
    retries
        Number of times to retry downloading before failing. Default: 6
    chunk_size
        Chunk size for downloading and unzipping. Default: 1024
    """
    def __init__(self,
                 folder: str,
                 model_path: str,
                 url: str,
                 retries: int = 6,
                 chunk_size: int = 1024) -> None:
        logger.debug(parse_class_init(locals()))
        self._cache_dir = folder
        self._model_path = model_path
        self._url = url
        self._retries = retries
        self._chunk_size = chunk_size

    @property
    def _model_zip_path(self) -> str:
        """ The full path to downloaded zip file. """
        retval = os.path.join(self._cache_dir, f"{self._model_path}.zip")
        logger.trace("[GetWeights] zip path: %s", repr(retval))  # type:ignore[attr-defined]
        return retval

    @property
    def _url_partial_size(self) -> int:
        """ How many bytes have already been downloaded. """
        zip_file = self._model_zip_path
        retval = os.path.getsize(zip_file) if os.path.exists(zip_file) else 0
        logger.trace("[GetWeights] Partial size: %s", retval)  # type:ignore[attr-defined]
        return retval

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
            logger.info("Zip already exists. Skipping download")
            return
        write_type = "wb" if downloaded_size == 0 else "ab"
        assert tqdm is not None
        with open(self._model_zip_path, write_type) as out_file:
            p_bar = tqdm(desc="Downloading",
                         unit="B",
                         total=length,
                         unit_scale=True,
                         unit_divisor=1024)
            if downloaded_size != 0:
                p_bar.update(downloaded_size)
            while True:
                buffer = response.read(self._chunk_size)
                if not buffer:
                    break
                p_bar.update(len(buffer))
                out_file.write(buffer)
            p_bar.close()

    def _download_model(self) -> None:
        """ Download the model zip from github to the cache folder. """
        logger.info("Downloading model: '%s' from: %s",
                    os.path.basename(self._model_path), self._url)
        for attempt in range(self._retries):
            try:
                downloaded_size = self._url_partial_size
                req = request.Request(self._url)
                if downloaded_size != 0:
                    req.add_header("Range", f"bytes={downloaded_size}-")
                with request.urlopen(req, timeout=10) as response:
                    logger.debug("[GetWeights] header info: {%s}", response.info())
                    logger.debug("[GetWeights] Return Code: %s", response.getcode())
                    self._write_zipfile(response, downloaded_size)
                break
            except (socket_error, socket_timeout,
                    urlliberror.HTTPError, urlliberror.URLError) as err:
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
                     unit_divisor=1024)
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
        logger.info("Extracting: '%s'", os.path.basename(self._model_path))
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


class GithubDownloader(Downloader):
    """ Download a model from deepfakes-models releases and unzip to the cache path.

    Parameters
    ----------
    folder
        The directory to cache the model in
    model_name
        The full path to the extracted location
    git_model_id
        The second digit in the github tag that identifies this model. See
        https://github.com/deepfakes-models/faceswap-models for more information
    retries
        Number of times to retry downloading before failing. Default: 6
    chunk_size
        Chunk size for downloading and unzipping. Default: 1024

    Notes
    ------
    Models must have a certain naming convention: `<model_name>_v<version_number>.<extension>`
    (eg: `s3fd_v1.pb`).

    Multiple models can exist within the model_filename. They should be passed as a list and follow
    the same naming convention as above. Any differences in filename should occur AFTER the version
    number: `<model_name>_v<version_number><differentiating_information>.<extension>` (eg:
    `["mtcnn_det_v1.1.py", "mtcnn_det_v1.2.py", "mtcnn_det_v1.3.py"]`, `["resnet_ssd_v1.caffemodel"
    ,"resnet_ssd_v1.prototext"]`

    Example
    -------
    >>> from lib.utils import GetWeights
    >>> model_downloader = GetWeights("s3fd_keras_v2.h5", 11)
    """
    def __init__(self,
                 folder: str,
                 model_path: str,
                 git_model_id: int,
                 retries=6,
                 chunk_size=1024) -> None:
        logger.debug(parse_class_init(locals()))

        url_base = "https://github.com/deepfakes-models/faceswap-models/releases/download"

        version = int(model_path[model_path.rfind("_") + 2:])
        tag = f"v{git_model_id}.{version}"

        url = f"{url_base}/{tag}/{model_path}.zip"

        super().__init__(folder, model_path, url, retries=retries, chunk_size=chunk_size)


__all__ = get_module_objects(__name__)


if __name__ == "__main__":
    m = GetWeights(
        model_filename=["mtcnn_det_v3.1.pt", "mtcnn_det_v3.2.pt", "mtcnn_det_v3.3.pt"],
        git_model_id=2)
    print(m.model_path)
