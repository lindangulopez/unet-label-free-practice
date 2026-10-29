"""Download one study of the Light My Cells database from the BioImage Archive (S-BIAD1047).

Usage: python download_study.py 2            # Study_2, about 140 MB
       python download_study.py 23 --max-sets 6

Files go to data/Study_<n>/. Files already there with the right size are skipped, so the
script can be re-run after an interruption; incomplete files are downloaded again.
"""
import argparse
import http.client
import re
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/047/S-BIAD1047/Files/Images"
HEADERS = {"User-Agent": "unet-label-free-practice/0.1 (learning exercise)"}


def fetch(url, dest=None, tries=5):
    """GET a URL, with a pause and retry when the server refuses (403/429/5xx)."""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120) as r:
                if dest is None:
                    return r.read().decode()
                tmp = dest.with_suffix(".part")
                with open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f)
                tmp.rename(dest)
                return None
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError) as e:
            retryable = not isinstance(e, urllib.error.HTTPError) or e.code in (403, 429, 500, 502, 503)
            if not retryable or attempt == tries - 1:
                raise
            print(f"  retry in {5 * 2 ** attempt} s ({e})")
            time.sleep(5 * 2 ** attempt)


def remote_size(url):
    req = urllib.request.Request(url, headers=HEADERS, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def list_files(study):
    html = fetch(f"{BASE}/Study_{study}/")
    return sorted(set(re.findall(r'href="(image_\d+_[^"]+\.ome\.tiff)"', html)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study", type=int, help="study number, 1-30")
    ap.add_argument("--max-sets", type=int, default=None, help="only the first N acquisition sets")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    files = list_files(args.study)
    if args.max_sets:
        keep = sorted({int(f.split("_")[1]) for f in files})[: args.max_sets]
        files = [f for f in files if int(f.split("_")[1]) in keep]

    out = Path(args.out) / f"Study_{args.study}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Study_{args.study}: {len(files)} files -> {out}")
    for i, name in enumerate(files, 1):
        dest = out / name
        if dest.exists() and dest.stat().st_size == remote_size(f"{BASE}/Study_{args.study}/{name}"):
            continue
        fetch(f"{BASE}/Study_{args.study}/{name}", dest)
        print(f"  {i}/{len(files)} {name}")
        time.sleep(0.2)  # be gentle with the archive


if __name__ == "__main__":
    main()
