"""Serve local Qwen weights through Transformers' OpenAI-compatible server."""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("D:/Models/Qwen3.5-4B"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--torch-site-packages", type=Path,
        default=Path(sys.prefix).parent / "graphrag/Lib/site-packages",
        help="Existing CUDA PyTorch and torchvision installation.",
    )
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    if not (args.model / "config.json").is_file():
        parser.error(f"Model configuration missing from {args.model}")
    packages = args.torch_site_packages.resolve()
    if not (packages / "torch").is_dir() or not (packages / "torchvision").is_dir():
        parser.error(f"CUDA PyTorch and torchvision missing from {packages}")

    # Import the matching native backends before using the active Transformers version.
    sys.path.insert(0, str(packages))
    try:
        import torch
        import torchvision
    finally:
        sys.path.remove(str(packages))
    if not torch.cuda.is_available():
        raise RuntimeError("The selected PyTorch installation cannot access CUDA")

    from transformers.cli.serve import Serve

    Serve(
        force_model=args.model.resolve().as_posix(), device="cuda:0", dtype="bfloat16",
        reasoning="off", host="127.0.0.1", port=args.port, log_level="info",
    )


if __name__ == "__main__":
    main()
