#!/usr/bin/env python3
"""Run one resize+JPEG config through the existing protocol-v2 runner."""

from __future__ import annotations

import argparse
import pathlib
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path, required=True)
    parser.add_argument("--scale", type=int, required=True,
                        help="Integer downscale divisor used by encode_context_only.")
    parser.add_argument("--quality", type=int, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gt", default="")
    parser.add_argument("--img-dir", default="")
    args = parser.parse_args()
    if args.scale < 1:
        parser.error("--scale must be >= 1")
    if not 1 <= args.quality <= 100:
        parser.error("--quality must be in 1..100")

    repo_root = args.repo_root.resolve()
    if not repo_root.is_dir():
        parser.error(f"repo root does not exist: {repo_root}")
    sys.path.insert(0, str(repo_root))
    from protocol_v2 import run_a

    config_id = f"RESIZE_S{args.scale}_Q{args.quality}"
    run_a.CONFIGS[config_id] = {
        "desc": f"Resize+JPEG context baseline, scale={args.scale}, q={args.quality}",
        "mode": "ctx", "scale": args.scale, "q": args.quality,
    }
    forwarded = ["--config", config_id, "--url", args.url, "--run-id", args.run_id]
    if args.gt:
        forwarded += ["--gt", args.gt]
    if args.img_dir:
        forwarded += ["--img-dir", args.img_dir]
    sys.argv = ["protocol_v2.run_a"] + forwarded
    run_a.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
