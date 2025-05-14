import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

TEMPORAL_RESPONSE_SUFFIX = (
    "Provide the sequence of letters that represents the correct temporal order. "
    "For example: (A)(B)(C)(D). "
    "Do not include any other text."
)

FOUR_CHOICES_SUFFIX = (
    "Answer only with the letter corresponding to your choice in parenthesis: "
    "(A), (B), (C) or (D). "
    "Do not include any other text."
)

FIVE_CHOICES_SUFFIX = (
    "Answer only with the letter corresponding to your choice in parenthesis: "
    "(A), (B), (C), (D) or (E). "
    "Do not include any other text."
)


class DatasetWrapper:

    def __init__(
        self,
        data_samples: List[Dict[str, Any]],
        prompt_type: str,
        data_dir_path: str,
    ):
        self.data_samples = data_samples
        self.prompt_type = prompt_type
        self.data_dir_path = Path(data_dir_path)

    def get_raw_choices(self, sample: Dict[str, Any]):
        raw_choices = sample[f"raw_choices_{self.prompt_type}"]
        if self.prompt_type not in ["multimodal", "silent_video", "audio", "text_only"]:
            if len(raw_choices) == 5:
                raw_choices = raw_choices[:-1]

        return raw_choices

    def __getitem__(self, idx: int):
        sample = self.huggingface_dataset[idx]
        file_path = self.get_file_paths(sample, prompt_type=self.prompt_type)
        raw_choices = self.get_raw_choices(sample=sample)
        ground_truth = self.get_ground_truth(
            prompt_type=self.prompt_type,
            raw_choices=raw_choices,
            sample=sample,
        )

        returned_sound_name = (
            None
            if self.prompt_type in ["temporal_video", "video_segment"]
            else sample["audio_class"].replace("_", " ")
        )

        return {
            "sample_id": self.get_sample_id(file_path),
            "file_path": file_path,
            "raw_choices": raw_choices,
            "org_narrations": [e["narration"] for e in sample.get("events", None)],
            "ground_truth": ground_truth,
            "sound_name": returned_sound_name,
            "prompt_type": self.prompt_type,
            # TODO: fix timestamps for none_of_the_above questions, should be None?
            "ground_truth_timestamps": self.get_gt_timestamps(sample),
            "type": sample.get("type", "regular"),
            "overlayed_event_index": sample.get("overlayed_event_index", None),
        }

    def __len__(self):
        return len(self.data_samples)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def get_sample_id(self, file_path: Union[Path, Dict[str, Path]]) -> str:
        if isinstance(file_path, dict):
            file_path = f"{str(file_path['video'])}"
        return f"{str(file_path)}+{self.prompt_type}"

    @property
    def prompt_suffix(self):
        return (
            TEMPORAL_RESPONSE_SUFFIX
            if self.prompt_type == "temporal_video"
            else (
                FOUR_CHOICES_SUFFIX
                if self.prompt_type
                in [
                    "video_segment",
                    "simple_audio_classification",
                    "overlayed_full_audio_classification",
                ]
                else FIVE_CHOICES_SUFFIX
            )
        )

    def get_file_paths(
        self, sample: Dict[str, Any], prompt_type: str
    ) -> Union[Path, Dict[str, Path]]:
        """Determine the correct file path based on prompt type."""
        path_mapping = {
            "multimodal": {
                "video": "video_with_overlayed_audio_path",
                "audio": "overlayed_audio_path",
            },
            "audio": "overlayed_audio_path",
            "silent_video": "silent_video_path",
            "text_only": None,  # NOTE: This will be ignored when we load the file later on
            "overlayed_full_audio_classification": "overlayed_audio_path",
            "simple_audio_classification": "audio_path",
            "pipeline_event_classification": "video_with_overlayed_audio_path",
            "temporal_video": "video_with_overlayed_audio_path",
            "video_segment": "event_video_path",
        }
        mapping = path_mapping[prompt_type]
        if isinstance(mapping, dict):
            file_path = {k: Path(sample[v]) for k, v in mapping.items()}
            # Doing this because the file paths are relative to the data directory
            file_path = {
                k: self.data_dir_path / f_path.relative_to("data")
                for k, f_path in file_path.items()
            }
        elif mapping is None:
            file_path = None
        else:
            file_path = Path(sample[mapping])
            # Doing this because the file paths are relative to the data directory
            file_path = self.data_dir_path / file_path.relative_to("data")

        return file_path

    def remove_ordinal_suffix(self, text: str) -> str:
        if ":" not in text:
            return text
        return text.split(":")[1].strip()

    def get_ground_truth(
        self,
        *,
        prompt_type: str,
        raw_choices: List[str],
        sample: Dict[str, Any],
    ) -> List[str]:
        if (
            prompt_type == "simple_audio_classification"
            or prompt_type == "overlayed_full_audio_classification"
        ):
            # Find the index of the ground truth class in the shuffled choices
            ground_truth_class = sample["audio_class"].replace("_", " ")
            correct_index = raw_choices.index(ground_truth_class)
            return [f"({chr(65 + correct_index)})"]
        elif prompt_type == "temporal_video":
            return sample["correct_temporal_order"]
        elif prompt_type == "video_segment":
            # overlayed_event_index is re-mapped to point to the "multimodal" group of prompts
            correct_event = sample["raw_choices_multimodal"][sample["overlayed_event_index"]]
            correct_event = self.remove_ordinal_suffix(correct_event)
            correct_indices = [
                i for i, choice in enumerate(raw_choices) if choice.lower() == correct_event
            ]
            return [f"({chr(65 + i)})" for i in correct_indices]
        elif prompt_type in [
            "multimodal",
            "silent_video",
            "audio",
            "text_only",
            "pipeline_event_classification",
        ]:
            # Calculate ground truth; adjusted for the None of the above option (no changes, as None of the above
            # is always last)
            if sample["type"] in [
                "none_of_the_above_no_sound",
                "none_of_the_above_incorrect_audio",
            ]:
                correct_index = len(raw_choices) - 1
            else:
                correct_index = sample["overlayed_event_index"]
            return [f"({chr(65 + correct_index)})"]

        raise ValueError(f"Unknown prompt type: {prompt_type}")

    def get_gt_timestamps(self, sample):
        def format_to_mm_ss(timestamp_str):
            time_obj = datetime.strptime(timestamp_str, "%H:%M:%S.%f")
            return time_obj.strftime("%M:%S")

        if sample["overlayed_event_index"] is None:
            return None, None
        gt_start = sample["events"][sample["overlayed_event_index"]]["start"]
        gt_end = sample["events"][sample["overlayed_event_index"]]["end"]

        gt_start = format_to_mm_ss(gt_start)
        gt_end = format_to_mm_ss(gt_end)

        gt_seconds_start, gt_seconds_end = self.extract_timestamps(f"[{gt_start}, {gt_end}]")

        return gt_seconds_start, gt_seconds_end

    @staticmethod
    def extract_timestamps(timestamps_string):
        """
        Extracts timestamps in seconds, handling:
        - MM:SS format (e.g., [00:05, 00:08])
        - HH:MM:SS format (e.g., [00:00:00, 00:00:03])
        - Floating-point seconds (e.g., [0.0, 5.0])

        Args:
            timestamps_string (str): The string containing the timestamps to be converted in seconds.

        Returns:
            tuple: (timestamp_start, timestamp_end) as (float, float)
        """
        # Normalize input by stripping square brackets if present
        timestamps_string = timestamps_string.strip("[]")

        # Regex to match two timestamps in different formats
        match = re.match(r"\s*([\d:.]+)\s*,\s*([\d:.]+)\s*", timestamps_string)

        if match:
            start_time_str = match.group(1)
            end_time_str = match.group(2)

            # Function to convert time string to seconds
            def time_to_seconds(time_str):
                if ":" in time_str:  # Convert HH:MM:SS or MM:SS to seconds
                    parts = list(map(float, time_str.split(":")))
                    return sum(part * (60**i) for i, part in enumerate(reversed(parts)))
                else:  # Already in seconds (float)
                    return float(time_str)

            timestamp_start = time_to_seconds(start_time_str)
            timestamp_end = time_to_seconds(end_time_str)

            return timestamp_start, timestamp_end

        else:
            raise ValueError(f"Invalid format for timestamps conversion: {timestamps_string}")
