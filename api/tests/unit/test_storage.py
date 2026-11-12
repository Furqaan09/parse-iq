from urllib.parse import parse_qs, urlparse

import pytest
from botocore.exceptions import ClientError
from moto import mock_aws
from pydantic import ValidationError

from app.settings import Settings
from app.storage import PRESIGNED_URL_SECONDS, LocalStorage, S3Storage

BUCKET = "parseiq-test-bucket"
KEY = "2026/11/abc123_invoice.pdf"


@pytest.fixture
def s3():
    with mock_aws():
        storage = S3Storage(
            bucket=BUCKET, region="us-east-1", access_key_id="testing", secret_access_key="testing"
        )
        storage.client.create_bucket(Bucket=BUCKET)
        yield storage


def test_s3_put_then_get_returns_the_same_bytes_and_content_type(s3):
    s3.put(KEY, b"%PDF-1.7 fake", "application/pdf")

    assert s3.get(KEY) == b"%PDF-1.7 fake"
    head = s3.client.head_object(Bucket=BUCKET, Key=KEY)
    assert head["ContentType"] == "application/pdf"


def test_s3_get_of_a_missing_key_raises(s3):
    with pytest.raises(ClientError):
        s3.get("2026/11/missing.pdf")


def test_s3_url_is_presigned_and_expires(s3):
    url = urlparse(s3.url(KEY))
    query = parse_qs(url.query)

    assert url.scheme == "https"
    assert BUCKET in url.netloc + url.path
    assert url.path.endswith(KEY)
    assert query["X-Amz-Expires"] == [str(PRESIGNED_URL_SECONDS)]
    assert "X-Amz-Signature" in query


def test_local_put_then_get_and_url(tmp_path):
    storage = LocalStorage(tmp_path)
    storage.put("2026/11/notes.txt", b"hello", "text/plain")

    assert storage.get("2026/11/notes.txt") == b"hello"
    assert storage.url("2026/11/notes.txt") == "/files/2026/11/notes.txt"


@pytest.mark.parametrize("key", ["../outside.txt", "2026/../../outside.txt", "/etc/passwd"])
def test_local_rejects_keys_outside_its_folder(tmp_path, key):
    storage = LocalStorage(tmp_path / "uploads")

    with pytest.raises(ValueError):
        storage.put(key, b"x", "text/plain")


def test_s3_backend_without_bucket_and_keys_fails_at_startup():
    with pytest.raises(ValidationError, match="S3_BUCKET"):
        Settings(_env_file=None, database_url="postgresql://unused", storage_backend="s3")
