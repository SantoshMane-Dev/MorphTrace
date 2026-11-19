#!/usr/bin/env python3
"""
Analyze an image with DeepAI's AI Image Detector, replicating what the
website does (reverse-engineered from a HAR capture).

Flow seen in the HAR:
  1. POST {API}/ai-image-detector-upload   multipart field "image"  -> {"id": "...", "status": "pending"}
  2. POST {API}/ai-image-detector-query    JSON {"id": "..."}       -> {"status": "pending"|"done", "result": ..., "result_details": {...}}
     (repeat step 2 until status == "done")

Usage:
    python deepai_detect.py photo.jpg
    python deepai_detect.py a.jpg b.png --json
"""
import argparse
import json
import mimetypes
import sys
import time
from pathlib import Path

import requests

API = "https://api.deepai.org"
HEADERS = {
    # The site only sends these; no cookies / API key were present in the capture.
    "Origin": "https://deepai.org",
    "Referer": "https://deepai.org/",
    "Accept": "*/*",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/157.0.0.0 Safari/537.36"
    ),
}


def upload(session: requests.Session, path: Path, api: str = API) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    with path.open("rb") as f:
        r = session.post(
            f"{api}/ai-image-detector-upload",
            files={"image": (path.name, f, mime)},
            timeout=60,
        )
    r.raise_for_status()
    data = r.json()
    if "id" not in data:
        raise RuntimeError(f"Upload returned no job id: {data}")
    return data["id"]


def wait_for_result(session: requests.Session, job_id: str, api: str = API,
                    interval: float = 1.0, timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = session.post(f"{api}/ai-image-detector-query", json={"id": job_id}, timeout=30)
        r.raise_for_status()
        data = r.json()
        status = data.get("status")
        if status == "done":
            return data
        err = (data.get("result_details") or {}).get("error_message")
        if status in ("error", "failed", "failure", "invalid") or err:
            raise RuntimeError(f"Job failed (status={status!r}, error={err!r})")
        # Anything else ("pending", "analyzing", ...) means still running.
        time.sleep(interval)
    raise TimeoutError(f"Job {job_id} not finished after {timeout}s")


def analyze(path: str, api: str = API) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    with requests.Session() as s:
        s.headers.update(HEADERS)
        job_id = upload(s, p, api)
        return wait_for_result(s, job_id, api)


def summarize(resp: dict) -> str:
    d = resp.get("result_details") or {}
    lines = [
        f"Verdict           : {d.get('final_result')}",
        f"Label confidence  : {d.get('final_label_confidence')}",
        f"ML model          : {d.get('ml_model')}",
        f"Metadata check    : {d.get('metadata')}",
        f"OCR check         : {d.get('ocr')}",
        f"SynthID check     : {d.get('synthid')}",
    ]
    for w in d.get("warnings") or []:
        lines.append(f"Warning           : {w.get('type')} -> {w.get('label')} ({w.get('confidence')})")
    if d.get("error_message"):
        lines.append(f"Error             : {d['error_message']}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--json", action="store_true", help="print the raw JSON response")
    ap.add_argument("--api", default=API, help=argparse.SUPPRESS)  # for testing against a mock
    args = ap.parse_args()

    rc = 0
    for img in args.images:
        try:
            resp = analyze(img, args.api)
        except Exception as e:  # keep going on multi-file runs
            print(f"[{img}] failed: {e}", file=sys.stderr)
            rc = 1
            continue
        print(f"== {img}")
        print(json.dumps(resp, indent=2) if args.json else summarize(resp))
    return rc


if __name__ == "__main__":
    sys.exit(main())
