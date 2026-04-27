#!/usr/bin/env python3
"""Download a trained Ximilar VLM model artifact by UUID."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from client import download_and_extract_model, get_required_env


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a trained Ximilar VLM model by UUID.")
    parser.add_argument("--model_uuid", required=True, help="UUID of the trained VLM model")
    parser.add_argument("--output_path", required=True, help="Directory where the model should be extracted")
    return parser


def main() -> None:
    args = get_arg_parser().parse_args()
    api_token = get_required_env("XIMILAR_API_TOKEN")
    api_url = get_required_env("XIMILAR_API_URL")
    output_path = Path(args.output_path).expanduser().resolve()
    result = download_and_extract_model(
        model_uuid=args.model_uuid,
        output_path=output_path,
        api_url=api_url,
        api_token=api_token,
    )
    print(result)


if __name__ == "__main__":
    main()
