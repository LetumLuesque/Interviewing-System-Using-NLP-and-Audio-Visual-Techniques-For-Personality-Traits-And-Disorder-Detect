"""
build_course_db.py
──────────────────
Validates every URL in edx_courses.json by sending a lightweight HTTP HEAD
request (with a browser-like User-Agent to avoid 403s).

Run once before training to confirm all course links are live.

Usage:
    python build_course_db.py
"""

import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("[ERROR] 'requests' is not installed. Run: pip install requests")
    sys.exit(1)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
COURSES_FILE = BASE_DIR / "edx_courses.json"

# ── HTTP settings ─────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
TIMEOUT = 10          # seconds per request
DELAY   = 0.5         # polite delay between requests (seconds)


def check_url(url: str) -> tuple[bool, int]:
    """Returns (is_live, status_code). Treats 2xx and 3xx as live."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT,
                            allow_redirects=True, stream=True)
        return (resp.status_code < 400, resp.status_code)
    except requests.RequestException as exc:
        print(f"    [WARN] Request failed for {url!r}: {exc}")
        return (False, 0)


def main():
    if not COURSES_FILE.exists():
        print(f"[ERROR] Course file not found: {COURSES_FILE}")
        sys.exit(1)

    with open(COURSES_FILE, encoding="utf-8") as f:
        courses = json.load(f)

    total       = len(courses)
    broken      = []
    by_category: dict[str, list] = {}

    print(f"\n{'='*60}")
    print(f"  edX Course Database Validator")
    print(f"  Checking {total} courses …")
    print(f"{'='*60}\n")

    for i, course in enumerate(courses, 1):
        cat   = course.get("category", "unknown")
        title = course.get("title", "No title")
        url   = course.get("url", "")

        by_category.setdefault(cat, []).append(course)

        print(f"[{i:02d}/{total}] {title}")
        print(f"         {url}")

        is_live, status = check_url(url)
        status_label = f"HTTP {status}" if status else "FAILED"
        result_label = "OK" if is_live else "BROKEN"
        print(f"         >> {result_label}  ({status_label})\n")

        if not is_live:
            broken.append({"title": title, "url": url, "status": status})

        time.sleep(DELAY)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Total courses  : {total}")
    print(f"  Live / OK      : {total - len(broken)}")
    print(f"  Broken links   : {len(broken)}")
    print()

    print("  Coverage by Category:")
    for cat, items in sorted(by_category.items()):
        print(f"    {cat:<20} {len(items)} course(s)")

    if broken:
        print(f"\n  Broken URLs to fix:")
        for b in broken:
            print(f"    [{b['status']}] {b['title']}")
            print(f"          {b['url']}")
        print()
        print("[RESULT] Some URLs need updating in edx_courses.json before training.")
        sys.exit(1)
    else:
        print(f"\n[RESULT] All {total} course URLs are live. Ready for training.")


if __name__ == "__main__":
    main()
