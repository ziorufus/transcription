# Add imports for the new functions
import json
import mimetypes
import os
import urllib.request
from pathlib import Path
from uuid import uuid4

def build_url(base_url: str, endpoint: str) -> str:
    return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"


def make_headers(token: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {token}"}
    if extra:
        headers.update(extra)
    return headers


def encode_multipart_formdata(fields: dict[str, str], file_field: str, file_path: Path):
    boundary = f"----SHASBoundary{uuid4().hex}"
    body = bytearray()
    crlf = b"\r\n"

    for name, value in fields.items():
        body.extend(f"--{boundary}".encode())
        body.extend(crlf)
        body.extend(
            f'Content-Disposition: form-data; name="{name}"'.encode()
        )
        body.extend(crlf)
        body.extend(crlf)
        body.extend(str(value).encode())
        body.extend(crlf)

    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    body.extend(f"--{boundary}".encode())
    body.extend(crlf)
    body.extend(
        (
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{file_path.name}"'
        ).encode()
    )
    body.extend(crlf)
    body.extend(f"Content-Type: {mime_type}".encode())
    body.extend(crlf)
    body.extend(crlf)
    body.extend(file_path.read_bytes())
    body.extend(crlf)
    body.extend(f"--{boundary}--".encode())
    body.extend(crlf)

    content_type = f"multipart/form-data; boundary={boundary}"
    return bytes(body), content_type


def request_json(
    url: str,
    method: str,
    token: str,
    data: bytes | None = None,
    content_type: str | None = None,
) -> dict:
    headers = make_headers(token)
    if content_type:
        headers["Content-Type"] = content_type

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        details = exc.read().decode(errors="replace")
        raise RuntimeError(f"{method} {url} failed: HTTP {exc.code} {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc


def download_file(url: str, token: str, output_path: Path) -> None:
    request = urllib.request.Request(url, headers=make_headers(token), method="GET")
    try:
        with urllib.request.urlopen(request) as response:
            output_path.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        details = exc.read().decode(errors="replace")
        raise RuntimeError(f"GET {url} failed: HTTP {exc.code} {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GET {url} failed: {exc.reason}") from exc