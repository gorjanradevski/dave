import argparse
import json
import time
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from datasets import load_dataset
from tqdm.auto import tqdm

from utils.constants import GOOGLE_API_KEY


def get_uploaded_files():
    """Fetches a dictionary of already uploaded files {file_name: file_id}."""
    uploaded_files = {}
    total_files = len([f for f in genai.list_files()])
    for file in tqdm(genai.list_files(), total=total_files):
        uploaded_files[file.display_name] = file.name  # Map local filename to file ID
    return uploaded_files


def upload_file(path: Path, uploaded_files: dict) -> Optional[str]:
    """Upload a file if not already uploaded and return its ID."""
    if not path.exists():
        print(f"File not found: {path}")
        return None

    # Check if file already exists
    if path.name in uploaded_files:
        return uploaded_files[path.name]  # Return existing file ID

    max_retries = 3
    delay = 60

    for attempt in range(max_retries):
        try:
            genai_file = genai.upload_file(str(path))

            while genai_file.state.name == "PROCESSING":
                time.sleep(0.1)
                genai_file = genai.get_file(genai_file.name)

            if genai_file.state.name == "FAILED":
                print(f"Upload failed for: {path}")
                return None

            return genai_file.name

        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Operation failed: {str(e)}, retrying in {delay} seconds...")
            time.sleep(delay)

    return None


def main():
    parser = argparse.ArgumentParser(description="Upload dataset files to GenAI.")
    parser.add_argument("--split", type=Path, required=True, help="The dataset split.")
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to save the file mapping JSON",
    )
    args = parser.parse_args()

    genai.configure(api_key=GOOGLE_API_KEY)

    # Fetch existing uploaded files
    print("Fetching existing uploaded files...")
    uploaded_files = get_uploaded_files()
    huggingface_dataset = load_dataset(
        "gorjanradevski/dave",
        split=args.split,
        keep_in_memory=True,
        trust_remote_code=True,
    )
    file_mapping = {}

    print("Uploading files...")
    for sample in tqdm(huggingface_dataset):
        for file_type in [
            "video_with_overlayed_audio_path",
            "overlayed_audio_path",
            "silent_video_path",
            "audio_path",
            "event_video_path",
            "compressed_video_path",
        ]:
            if file_type not in sample:
                continue
            file_path = Path(sample[file_type])
            if str(file_path) in file_mapping:
                continue

            file_id = upload_file(file_path, uploaded_files)
            if file_id:
                file_mapping[str(file_path)] = file_id
                uploaded_files[file_path.name] = file_id  # Update cache

    json.dump(file_mapping, open(args.output_path, "w"))
    print(f"File mapping saved to {args.output_path}")


if __name__ == "__main__":
    main()
