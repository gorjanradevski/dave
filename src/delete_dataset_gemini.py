import argparse
import concurrent.futures
import json
from pathlib import Path

import google.generativeai as genai
from google.api_core.exceptions import PermissionDenied
from tqdm.auto import tqdm

from utils.constants import GOOGLE_API_KEY


def delete_file(google_file_id):
    try:
        genai.delete_file(google_file_id)
    except PermissionDenied:
        pass


def main():
    parser = argparse.ArgumentParser(description="Delete dataset files from GenAI.")
    parser.add_argument(
        "--file_mapping_path", type=Path, required=True, help="Path to the dataset mapping file"
    )
    parser.add_argument(
        "--num_threads", type=int, default=8, help="Number of threads to use for deletion"
    )
    args = parser.parse_args()

    genai.configure(api_key=GOOGLE_API_KEY)

    # Fetch existing uploaded files
    file_mapping = json.load(open(args.file_mapping_path))
    file_mapping_values = list(reversed(file_mapping.values()))  # Delete in reverse

    print("Deleting files...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = {
            executor.submit(delete_file, file_id): file_id for file_id in file_mapping_values
        }
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(file_mapping_values)):
            pass


if __name__ == "__main__":
    main()
