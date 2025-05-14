import argparse
import json
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from utils.config import DatasetConfig
from utils.dataset_utils import VideosForParticipant
from utils.processing_utils import AudioProcessor, VideoProcessor

# Suppress specific UserWarnings from moviepy
warnings.filterwarnings("ignore", category=UserWarning, module="moviepy")


class ParticipantProcessor:

    def __init__(
        self,
        read_data_path: Path,
        write_data_path: Path,
        config: DatasetConfig,
        video_processor: VideoProcessor,
        audio_processor: AudioProcessor,
    ):
        self.read_data_path = read_data_path
        self.write_data_path = write_data_path
        self.config = config
        self.video_processor = video_processor
        self.audio_processor = audio_processor

    def __call__(self, participant_id: str):
        video_directory_path = (
            self.write_data_path
            / self.config.PROCESSED_VIDEOS_DIR_NAME
            / f"{participant_id}_processed"
        )
        if not video_directory_path.exists():
            # If the video_directory_path starts with /readonly, remove it
            if str(video_directory_path).startswith("/readonly"):
                video_directory_path = video_directory_path.relative_to("/readonly")
            video_directory_path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"Folder {video_directory_path} already exists! Skipping {participant_id}")
            return []

        visual_csv_file_path = self.read_data_path / "EPIC-KITCHENS" / "EPIC_100_train.csv"
        videos_for_participant = VideosForParticipant.from_csv_file(
            visual_csv_file_path, participant_id
        )
        videos_for_participant.sort_event_list_by_start_time()
        videos_for_participant.merge_consequtive_events_same_action()
        videos_for_participant.convert_events_to_event_groups()
        videos_for_participant.filter_event_groups(config=self.config)
        videos_for_participant.flag_overlay_events(config=self.config)
        videos_for_participant.trim_and_save_videos(
            config=self.config,
            read_data_path=self.read_data_path,
            write_data_path=self.write_data_path,
        )
        samples_for_participant = videos_for_participant.create_samples_for_participant(
            video_processor=self.video_processor, audio_processor=self.audio_processor
        )

        return samples_for_participant


def compare_config(new_config, existing_config):
    """Compare two config dictionaries, ignoring EPIC_KITCHENS_PARTICIPANTS."""
    ignored_key = "PARTICIPANTS"

    # Create copies without the ignored key
    new_config_filtered = {k: v for k, v in new_config.items() if k != ignored_key}
    existing_config_filtered = {k: v for k, v in existing_config.items() if k != ignored_key}

    # Return the comparison result: Mismatched keys and values if not equal, empty dict otherwise
    mismatched_values = {
        k: {"new": new_config_filtered[k], "existing": existing_config_filtered[k]}
        for k in new_config_filtered.keys()
        if new_config_filtered[k] != existing_config_filtered[k]
    }

    return mismatched_values


def generate_dataset(args: argparse.Namespace):
    read_data_path, write_data_path = Path(args.read_data_path), Path(args.write_data_path)
    config = DatasetConfig.from_args(args)

    # Prepare and save once the configuration
    print("*" * 50)
    print("Configuration:")
    print(json.dumps(config.to_dict(), indent=4))
    print("*" * 50)

    # Check if the dataset already exists
    existing_dataset = None
    if Path(args.save_dataset_path).exists():
        existing_dataset = json.load(open(args.save_dataset_path))

    if existing_dataset:
        print("Existing dataset found. Checking for configuration changes...")
        existing_config = existing_dataset["meta"]["config"]

        # Check if the configuration has changed
        config_change = compare_config(new_config=config.to_dict(), existing_config=existing_config)
        if not config_change:
            new_participants = list(set(config.PARTICIPANTS) - set(existing_config["PARTICIPANTS"]))
            if new_participants:
                print(f"New participants detected: {new_participants}")
                config.PARTICIPANTS = existing_config["PARTICIPANTS"] + new_participants
            else:
                print("Configuration is identical. No changes needed.")
                return
        else:
            raise ValueError(
                f"Configuration mismatch detected in {args.save_dataset_path}; {config_change}"
            )

    # Save the dataset with the new configuration
    dataset = {
        "meta": {
            "config": config.to_dict(),
            "date": datetime.now().strftime("%Y-%m-%d"),
        },
        "data": existing_dataset["data"] if existing_dataset else [],
    }
    json.dump(dataset, open(args.save_dataset_path, "w"))

    # Prepare audio processor
    audio_processor = AudioProcessor(config=config, esc_path=read_data_path / "ESC-50-master")
    video_processor = VideoProcessor(config=config)
    participant_processor = ParticipantProcessor(
        read_data_path=read_data_path,
        write_data_path=write_data_path,
        config=config,
        video_processor=video_processor,
        audio_processor=audio_processor,
    )
    print("Processing participants...")

    # Locks for thread safety
    dataset_lock = threading.Lock()
    file_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Create the futures
        futures = {
            participant_id: executor.submit(participant_processor, participant_id)
            for participant_id in config.PARTICIPANTS
        }

        # Process results as they complete
        for participant_id in futures.keys():
            samples_for_participant = futures[participant_id].result()

            # Protect dataset modification
            with dataset_lock:
                dataset["data"].extend(samples_for_participant)

            # Safely save with a lock
            with file_lock:
                print(f"Saving after participant {participant_id}...")
                with open(args.save_dataset_path, "w") as f:
                    json.dump(dataset, f)

    print("Finished! Saving one last time!")
    time.sleep(5)

    # Final save with locks to ensure consistency
    with dataset_lock, file_lock:
        with open(args.save_dataset_path, "w") as f:
            json.dump(dataset, f)


def main():
    parser = argparse.ArgumentParser(description="Save videos with audio overlay")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument(
        "--read_data_path", type=str, help="Directory path from where to READ videos"
    )
    parser.add_argument(
        "--write_data_path", type=str, help="Directory path from where to WRITE videos"
    )
    parser.add_argument(
        "--save_dataset_path",
        type=str,
        default="data/epic_kitchens_dataset.json",
        help="Path where to save the dataset",
    )
    # Add arguments for each configurable parameter
    parser.add_argument(
        "--minimum_event_duration", type=float, help="Minimum duration for an event"
    )
    parser.add_argument(
        "--min_event_duration_for_overlay",
        type=float,
        help="The minimum event duration for overlay.",
    )
    parser.add_argument(
        "--audio_start_offset",
        type=float,
        help="For how much to offset the audio start so that it doesn't start at the previous event.",
    )
    parser.add_argument(
        "--words_to_filter", nargs="+", type=str, help="List of words to filter", default=["unsure"]
    )
    parser.add_argument("--max_overlap", type=float, help="Maximum allowed overlap duration")
    parser.add_argument("--sound_classes", nargs="+", type=str, help="List of sound classes")
    parser.add_argument("--max_sequence_duration", type=float, help="Maximum sequence duration")
    parser.add_argument("--audio_scale_coefficient", help="Audio scale coefficient", type=float)
    parser.add_argument("--fade_in_out_duration", type=float, help="Fade in/out duration")
    parser.add_argument("--participants", nargs="+", type=str, help="List of participants")
    parser.add_argument("--video_resize_percentage", type=float, help="Resize percentage for video")
    parser.add_argument(
        "--processed_videos_dir_name", type=str, help="Name of the processed videos directory"
    )
    args = parser.parse_args()
    generate_dataset(args)


if __name__ == "__main__":
    main()
