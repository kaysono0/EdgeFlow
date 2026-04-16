#!/usr/bin/env python3
"""
Download large model assets from a GitHub Release or any static file host.

Usage:
    python scripts/download_release_assets.py --base-url https://github.com/<owner>/<repo>/releases/download/<tag>

Or set:
    EDGEFLOW_RELEASE_BASE_URL=https://github.com/<owner>/<repo>/releases/download/<tag>
"""

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "release-assets.json"


def load_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["assets"]


def resolve_base_url(cli_base_url):
    base_url = cli_base_url or os.environ.get("EDGEFLOW_RELEASE_BASE_URL")
    if base_url:
        return base_url.rstrip("/")

    raise SystemExit(
        "Missing release base URL.\n"
        "Pass --base-url or set EDGEFLOW_RELEASE_BASE_URL.\n"
        "Example: https://github.com/<owner>/<repo>/releases/download/<tag>"
    )


def download_file(base_url, asset):
    target_path = REPO_ROOT / asset["path"]
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        print(f"✅ 已存在，跳过: {target_path}")
        return

    source_url = f"{base_url}/{asset['name']}"
    print(f"📥 下载 {asset['name']}")
    print(f"   来源: {source_url}")
    urllib.request.urlretrieve(source_url, str(target_path))
    print(f"   保存: {target_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", help="Release assets base URL")
    args = parser.parse_args()

    base_url = resolve_base_url(args.base_url)
    assets = load_manifest()

    failures = []
    for asset in assets:
        try:
            download_file(base_url, asset)
        except Exception as exc:
            failures.append((asset["name"], str(exc)))

    if failures:
        print("\n❌ 以下资产下载失败:")
        for name, error in failures:
            print(f"  - {name}: {error}")
        sys.exit(1)

    print("\n✅ Release 资产下载完成")


if __name__ == "__main__":
    main()
