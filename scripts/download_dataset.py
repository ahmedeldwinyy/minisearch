import argparse
import shutil
import ssl
from pathlib import Path
from urllib.request import urlopen

import certifi

DATASET_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
)
DEFAULT_DESTINATION = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "scifact.zip"
)


def download_dataset(destination: Path = DEFAULT_DESTINATION) -> Path:
    if destination.exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_destination = destination.with_suffix(f"{destination.suffix}.part")
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    try:
        with urlopen(DATASET_URL, timeout=60, context=ssl_context) as response:
            with temporary_destination.open("wb") as archive_file:
                shutil.copyfileobj(response, archive_file)
        temporary_destination.replace(destination)
    finally:
        temporary_destination.unlink(missing_ok=True)

    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the BEIR SciFact dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=f"Archive destination (default: {DEFAULT_DESTINATION})",
    )
    arguments = parser.parse_args()
    archive_path = download_dataset(arguments.output)
    print(f"SciFact archive available at {archive_path}")


if __name__ == "__main__":
    main()
