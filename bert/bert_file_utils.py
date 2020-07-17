# See source
# https://github.com/huggingface/transformers/blob/f82a2a5e8e6827343322a4a9831924c5bb9bd2b2/src/transformers/file_utils.py
# Modified!
import requests
from functools import partial
import tempfile
from tqdm import tqdm
import os
import json

WEIGHTS_NAME = "pytorch_model.bin"

CLOUDFRONT_DISTRIB_PREFIX = "https://cdn.huggingface.co"
S3_BUCKET_PREFIX = "https://s3.amazonaws.com/models.huggingface.co/bert"


def hf_bucket_url(model_id, filename, use_cdn=True):
    endpoint = CLOUDFRONT_DISTRIB_PREFIX if use_cdn else S3_BUCKET_PREFIX
    legacy_format = "/" not in model_id
    if legacy_format:
        return f"{endpoint}/{model_id}-{filename}"
    else:
        return f"{endpoint}/{model_id}/{filename}"


def is_url(path):
    return "https" in path


def download_file(url, temp_file):
    response = requests.get(url, stream=True)
    total_length = response.headers["Content-Length"]
    progress = tqdm(initial=0, total=int(total_length), desc="Downloading", )
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def cached_path(url,
                cache_dir=None,
                force_download=False, ):
    try:
        response = requests.head(url, allow_redirects=True)
        if response.status_code == 200:
            etag = response.headers.get("ETag")
    except RuntimeError:
        raise RuntimeError("Invalid URL")

    filename = url.split('/')[-1] + '.' + etag.replace("\"", "")

    cache_path = os.path.join(cache_dir, filename)

    if os.path.exists(cache_path) and not force_download:
        return cache_path
    else:
        temp_file_manager = partial(tempfile.NamedTemporaryFile, dir=cache_dir, delete=False)

    with temp_file_manager() as temp_file:
        download_file(url, temp_file)

    os.replace(temp_file.name, cache_path)

    meta = {"url": url, "etag": etag}
    meta_path = cache_path + '.json'
    with open(meta_path, 'w') as meta_file:
        json.dump(meta, meta_file)

    return cache_path


if __name__ == '__main__':
    url = 'https://cdn.huggingface.co/bert-base-uncased-pytorch_model.bin'
    resolved_archive_file = cached_path(url, cache_dir='./')
