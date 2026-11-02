"""Pin public benchmark PDFs in a dedicated directory; never fetch private sources."""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

from core.extract.classify import count_pages

PDFS = {
    "attention": "https://arxiv.org/pdf/1706.03762",
    "bert": "https://arxiv.org/pdf/1810.04805",
    "http-semantics": "https://www.rfc-editor.org/rfc/rfc9110.pdf",
    "udhr-french": (
        "https://www.ohchr.org/sites/default/files/UDHR/Documents/UDHR_Translations/frn.pdf"
    ),
    "udhr-arabic": "https://www.un.org/ar/udhrbook/pdf/udhr_booklet_ar_web.pdf",
    "wesp-2024": "https://desapublications.un.org/file/20347/download",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--pdf", action="append", choices=list(PDFS), help="Fetch selected fixtures"
    )
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    manifest_path = args.directory / "pdf-manifest.json"
    existing = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}
    for key, url in PDFS.items():
        if args.pdf and key not in args.pdf:
            continue
        path = args.directory / f"{key}.pdf"
        if key in existing:
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != existing[key]["sha256"]:
                raise ValueError("benchmark PDF bytes changed")
            print(key, existing[key]["pages"], existing[key]["sha256"])
            continue
        request = urllib.request.Request(url, headers={"User-Agent": "Corpi-public-benchmark/1"})
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=30) as response:
            data = bytearray()
            while block := response.read1(64 * 1024):
                data.extend(block)
                if len(data) > 25_000_000 or time.monotonic() - started > 120:
                    raise ValueError("benchmark PDF download limit")
        if not data.startswith(b"%PDF-"):
            raise ValueError("not a PDF")
        path.write_bytes(data)
        entry = {
            "url": url,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "pages": count_pages(bytes(data), stop_after=1001),
        }
        existing[key] = entry
        manifest_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        print(key, entry["pages"], entry["sha256"])


if __name__ == "__main__":
    main()
