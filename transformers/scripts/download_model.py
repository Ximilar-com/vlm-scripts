#!/usr/bin/env python3
"""
Download a trained Ximilar VLM model artifact by UUID.

RUN: uv run scripts/download_model.py --model-uuid <MODEL_UUID> --output-path <OUTPUT_PATH>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from client import download_and_extract_model, get_required_env


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a trained Ximilar VLM model by UUID.")
    parser.add_argument("--model-uuid", required=True, help="UUID of the trained VLM model")
    parser.add_argument("--output-path", required=True, help="Directory where the model should be extracted")
    return parser


def main() -> None:
    args = get_arg_parser().parse_args()
    print(f"[DEBUG] Parsed arguments: model_uuid={args.model_uuid} output_path={args.output_path}", file=sys.stderr)
    api_token = get_required_env("XIMILAR_API_TOKEN")
    api_url = get_required_env("XIMILAR_API_URL")
    print(f"[DEBUG] Loaded API configuration: api_url={api_url}", file=sys.stderr)
    output_path = Path(args.output_path).expanduser().resolve()
    print(f"[DEBUG] Resolved output path: {output_path}", file=sys.stderr)
    print(f"[DEBUG] Starting model download for {args.model_uuid}", file=sys.stderr)
    result = download_and_extract_model(
        model_uuid=args.model_uuid,
        output_path=output_path,
        api_url=api_url,
        api_token=api_token,
    )
    print(f"[DEBUG] Model download finished: {result}", file=sys.stderr)
    print(result)


if __name__ == "__main__":
    main()
