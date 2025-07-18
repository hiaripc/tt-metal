# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# Google utils: https://cloud.google.com/storage/docs/reference/libraries

import os
import platform
import subprocess
import time
from pathlib import Path

import requests
import torch
from loguru import logger


def gsutil_getsize(url=""):
    # gs://bucket/file size https://cloud.google.com/storage/docs/gsutil/commands/du
    s = subprocess.check_output(f"gsutil du {url}", shell=True).decode("utf-8")
    return eval(s.split(" ")[0]) if len(s) else 0  # bytes


def attempt_download(file, repo="WongKinYiu/yolov7"):
    # Attempt file download if does not exist
    file = Path(str(file).strip().replace("'", "").lower())

    if not file.exists():
        try:
            response = requests.get(f"https://api.github.com/repos/{repo}/releases/latest").json()  # github api
            assets = [x["name"] for x in response["assets"]]  # release assets
            tag = response["tag_name"]  # i.e. 'v1.0'
        except:  # fallback plan
            assets = [
                "yolov7.pt",
                "yolov7-tiny.pt",
                "yolov7x.pt",
                "yolov7-d6.pt",
                "yolov7-e6.pt",
                "yolov7-e6e.pt",
                "yolov7-w6.pt",
            ]
            tag = subprocess.check_output("git tag", shell=True).decode().split()[-1]

        name = file.name
        if name in assets:
            msg = f"{file} missing, try downloading from https://github.com/{repo}/releases/"
            redundant = False  # second download option
            try:  # GitHub
                url = f"https://github.com/{repo}/releases/download/{tag}/{name}"
                logger.info(f"Downloading {url} to {file}...")
                torch.hub.download_url_to_file(url, file)
                assert file.exists() and file.stat().st_size > 1e6  # check
            except Exception as e:  # GCP
                logger.info(f"Download error: {e}")
                assert redundant, "No secondary mirror"
                url = f"https://storage.googleapis.com/{repo}/ckpt/{name}"
                logger.info(f"Downloading {url} to {file}...")
                os.system(f"curl -L {url} -o {file}")  # torch.hub.download_url_to_file(url, weights)
            finally:
                if not file.exists() or file.stat().st_size < 1e6:  # check
                    file.unlink(missing_ok=True)  # remove partial downloads
                    logger.info(f"ERROR: Download failure: {msg}")
                logger.info("")
                return


def gdrive_download(id="", file="tmp.zip"):
    # Downloads a file from Google Drive. from yolov7.utils.google_utils import *; gdrive_download()
    t = time.time()
    file = Path(file)
    cookie = Path("cookie")  # gdrive cookie
    logger.info(
        f"Downloading https://drive.google.com/uc?export=download&id={id} as {file}... ",
        end="",
    )
    file.unlink(missing_ok=True)  # remove existing file
    cookie.unlink(missing_ok=True)  # remove existing cookie

    # Attempt file download
    out = "NUL" if platform.system() == "Windows" else "/dev/null"
    os.system(f'curl -c ./cookie -s -L "drive.google.com/uc?export=download&id={id}" > {out}')
    if os.path.exists("cookie"):  # large file
        s = f'curl -Lb ./cookie "drive.google.com/uc?export=download&confirm={get_token()}&id={id}" -o {file}'
    else:  # small file
        s = f'curl -s -L -o {file} "drive.google.com/uc?export=download&id={id}"'
    r = os.system(s)  # execute, capture return
    cookie.unlink(missing_ok=True)  # remove existing cookie

    # Error check
    if r != 0:
        file.unlink(missing_ok=True)  # remove partial
        logger.info("Download error ")  # raise Exception('Download error')
        return r

    # Unzip if archive
    if file.suffix == ".zip":
        logger.info("unzipping... ", end="")
        os.system(f"unzip -q {file}")  # unzip
        file.unlink()  # remove zip to free space

    logger.info(f"Done ({time.time() - t:.1f}s)")
    return r


def get_token(cookie="./cookie"):
    with open(cookie) as f:
        for line in f:
            if "download" in line:
                return line.split()[-1]
    return ""


# def upload_blob(bucket_name, source_file_name, destination_blob_name):
#     # Uploads a file to a bucket
#     # https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python
#
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
#
#     blob.upload_from_filename(source_file_name)
#
#     logger.info('File {} uploaded to {}.'.format(
#         source_file_name,
#         destination_blob_name))
#
#
# def download_blob(bucket_name, source_blob_name, destination_file_name):
#     # Uploads a blob from a bucket
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)
#
#     blob.download_to_filename(destination_file_name)
#
#     logger.info('Blob {} downloaded to {}.'.format(
#         source_blob_name,
#         destination_file_name))
