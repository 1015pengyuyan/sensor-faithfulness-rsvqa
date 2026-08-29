#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from sensor_faithfulness.data.render_bigearthnet import render_s1_files, render_s2_files


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="modality", required=True)
    s1 = subparsers.add_parser("s1")
    s1.add_argument("--vv", type=Path, required=True)
    s1.add_argument("--vh", type=Path, required=True)
    s1.add_argument("--output", type=Path, required=True)
    s2 = subparsers.add_parser("s2")
    s2.add_argument("--b02", type=Path, required=True)
    s2.add_argument("--b03", type=Path, required=True)
    s2.add_argument("--b04", type=Path, required=True)
    s2.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.modality == "s1":
        render_s1_files(args.vv, args.vh, args.output, 512)
    else:
        render_s2_files(args.b02, args.b03, args.b04, args.output, 512)


if __name__ == "__main__":
    main()
