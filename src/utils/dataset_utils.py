import csv
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["IMAGEIO_NO_FFMPEG_PRINT"] = "1"  # Suppress `imageio-ffmpeg` logs


import json
import re

import ffmpeg
from tqdm.auto import tqdm

from utils.config import DatasetConfig
from utils.constants import BASE_DATETIME
from utils.processing_utils import AudioProcessor, VideoProcessor

# Suppress specific UserWarnings from moviepy
warnings.filterwarnings("ignore", category=UserWarning, module="moviepy")


class Event:
    def __init__(
        self,
        start: str | float | timedelta,
        end: str | float | timedelta,
        narration: str,
        action: str,
        video_id: str,
        overlay: bool = False,
    ):
        # Convert start time
        if isinstance(start, str):
            self.start = datetime.strptime(start, "%H:%M:%S.%f")
        elif isinstance(start, float):
            # Convert seconds to timedelta and add to BASE_DATETIME
            self.start = BASE_DATETIME + timedelta(seconds=float(start))
        elif isinstance(start, timedelta):
            # If the start time is a timedelta, add it to the base datetime
            self.start = BASE_DATETIME + start
        else:
            raise ValueError(f"Start time {start} is not a valid type: {type(start)}")

        # Convert end time
        if isinstance(end, str):
            self.end = datetime.strptime(end, "%H:%M:%S.%f")
        elif isinstance(end, float):
            self.end = BASE_DATETIME + timedelta(seconds=float(end))
        elif isinstance(end, timedelta):
            # If the end time is a timedelta, add it to the BASE_DATETIME
            self.end = BASE_DATETIME + end
        else:
            raise ValueError(f"End time {end} is not a valid type: {type(end)}")

        self.duration = self.get_duration()
        self.narration = narration
        self.action = action
        self.video_id = video_id
        self.overlay = overlay

    def set_overlay(self, overlay: bool) -> None:
        self.overlay = overlay

    def is_overlay(self) -> bool:
        return self.overlay

    def __str__(self):
        return f"Start: {self.start}; End: {self.end}; Duration: {self.duration}; Narration: {self.narration}"

    def get_duration(self) -> float:
        if self.end < self.start:
            raise ValueError(f"End time ({self.end}) cannot be before start time ({self.start})")
        return (self.end - self.start).total_seconds()

    @staticmethod
    def from_epic_kitchens_row(row: dict) -> Optional["Event"]:
        try:
            return Event(
                start=row["start_timestamp"],
                end=row["stop_timestamp"],
                narration=row.get("narration", ""),
                action=f"{row['verb']} {row['noun']}",
                video_id=row["video_id"],
            )
        except KeyError:
            return None

    def overlaps_with(self, other: "Event", max_overlap: float) -> bool:
        overlap_duration = (self.end - other.start).total_seconds()
        return overlap_duration > max_overlap

    def adjust_time(self, offset: timedelta) -> "Event":
        adjusted_start = self.start - offset
        adjusted_end = self.end - offset

        return Event(
            start=adjusted_start,
            end=adjusted_end,
            narration=self.narration,
            action=self.action,
            video_id=self.video_id,
            overlay=self.overlay,
        )

    def merge_with(self, other: "Event") -> "Event":
        start = min(self.start, other.start)
        end = max(self.end, other.end)

        # Convert to seconds from a common reference point instead of using strftime
        start_seconds = (start - BASE_DATETIME).total_seconds()
        end_seconds = (end - BASE_DATETIME).total_seconds()

        return Event(
            start=start_seconds,
            end=end_seconds,
            narration=self.narration,
            action=self.action,
            video_id=self.video_id,
            overlay=self.overlay,
        )

    def to_dict(self) -> dict:
        return {
            "start": self.start.strftime("%H:%M:%S.%f"),
            "end": self.end.strftime("%H:%M:%S.%f"),
            "duration": self.duration,
            "narration": self.narration,
            "action": self.action,
        }


class EventGroup:
    # A group of four events which will constitute a single event group; this event group, if valid, will be a single
    # sample in the dataset
    def __init__(self, events: List[Event], video_path: Optional[Path] = None):
        assert len(events) == 4
        self.events = events
        self.video_path = video_path

    def get_event_group_as_dict(self) -> List[Dict[str, str]]:
        return [e.to_dict() for e in self.events]

    def has_video_path(self) -> bool:
        return self.video_path is not None

    def get_events_for_group(self) -> List[Event]:
        return self.events

    def __str__(self):
        events_str = "\n".join(str(e) for e in self.events)
        return events_str

    @property
    def video_id(self) -> str:
        return self.events[0].video_id

    @property
    def duration(self) -> float:
        return sum(e.duration for e in self.events)

    def valid_group_duration(self, max_duration: float) -> bool:
        return self.duration <= max_duration

    def valid_overlap(self, max_overlap: float) -> bool:
        for i in range(1, len(self.events)):
            prev_end_time = self.events[i - 1].end
            curr_start_time = self.events[i].start

            overlap_duration = (curr_start_time - prev_end_time).total_seconds()
            if abs(overlap_duration) > max_overlap:
                return False

        return True

    def valid_text(self, words_to_filter: List[str], enforse_unique_events: bool) -> bool:
        # Check for each of the words, and their variants: Capital first letter, all caps, all lowercas
        expanded_words_to_filter = []
        for word in words_to_filter:
            expanded_words_to_filter.extend([word, word.capitalize(), word.upper(), word.lower()])

        # Check if any of the words are in the narration
        for word in expanded_words_to_filter:
            if any(word in e.narration for e in self.events):
                return False

        # No event must be a sub event of another & check if the narrations are unique
        if enforse_unique_events:
            for i in range(len(self.events)):
                for j in range(i + 1, len(self.events)):
                    if (
                        self.events[i].narration in self.events[j].narration
                        or self.events[i].narration == self.events[j].narration
                    ):
                        return False

        return True

    def at_least_one_event_above_duration(self, min_duration: float) -> bool:
        return any(e.duration >= min_duration for e in self.events)

    def valid_event_durations(self, min_duration: float, max_duration: float) -> bool:
        return all(e.duration >= min_duration for e in self.events) and all(
            e.duration <= max_duration for e in self.events
        )

    def get_group_start_and_end_time(self) -> Tuple[datetime, datetime]:
        return self.events[0].start, self.events[-1].end

    def flag_overlay_events(self, config: DatasetConfig) -> None:
        for event in self.events:
            if event.duration >= config.MIN_EVENT_DURATION_FOR_OVERLAY:
                event.set_overlay(True)

    def adjust_event_times(self) -> None:
        offset = self.events[0].start
        self.events = [e.adjust_time(offset) for e in self.events]

    def set_video_path(self, video_path: Path):
        self.video_path = video_path

    def get_input_output_epic_path(
        self,
        read_data_path: Path,
        write_data_path: Path,
        processed_videos_dir_name: str,
        video_id: str,
        participant_id: str,
    ):
        start_time, end_time = self.get_group_start_and_end_time()
        formatted_time = f"{start_time:%M-%S}+{end_time:%M-%S}"

        input_video_path = (
            read_data_path / "EPIC-KITCHENS" / participant_id / "videos" / f"{video_id}.MP4"
        )
        output_video_path = (
            write_data_path
            / processed_videos_dir_name
            / f"{participant_id}_processed"
            / f"{video_id}+{formatted_time}.mp4"
        )

        return input_video_path, output_video_path

    def get_input_output_ego4d_path(
        self,
        read_data_path: Path,
        write_data_path: Path,
        processed_videos_dir_name: str,
        video_id: str,
    ):
        start_time, end_time = self.get_group_start_and_end_time()
        formatted_time = f"{start_time:%M-%S}+{end_time:%M-%S}"

        input_video_path = read_data_path / "ego4d" / "v2" / "video_540ss" / f"{video_id}.mp4"
        output_video_path = (
            write_data_path
            / processed_videos_dir_name
            / f"{video_id}_processed"
            / f"{video_id}+{formatted_time}.mp4"
        )

        return input_video_path, output_video_path

    def get_input_output_video_path(
        self,
        dataset_name: str,
        read_data_path: Path,
        write_data_path: Path,
        video_id: str,
        processed_videos_dir_name: str,
        **kwargs,
    ):
        if dataset_name == "epic":
            return self.get_input_output_epic_path(
                read_data_path=read_data_path,
                write_data_path=write_data_path,
                video_id=video_id,
                processed_videos_dir_name=processed_videos_dir_name,
                **kwargs,
            )
        elif dataset_name == "ego4d":
            return self.get_input_output_ego4d_path(
                read_data_path=read_data_path,
                write_data_path=write_data_path,
                video_id=video_id,
                processed_videos_dir_name=processed_videos_dir_name,
            )
        raise ValueError(f"Dataset name {dataset_name} not recognized!")

    def trim_and_save_video(
        self,
        input_video_path: Path,
        output_video_path: Path,
    ) -> Optional[Path]:
        start_time, _ = self.get_group_start_and_end_time()
        # Use ffmpeg-python to trim the video
        try:
            # datetime.datetime objects represent points in time, and only datetime.timedelta objects (which represent
            # durations) have the total_seconds method
            start_time_seconds = (start_time - BASE_DATETIME).total_seconds()
            ffmpeg.input(input_video_path, ss=start_time_seconds, t=self.duration).output(
                str(output_video_path),
                codec="copy",
                avoid_negative_ts="make_zero",
                force_key_frames="expr:gte(t,n_forced*1)",
                loglevel="quiet",
            ).run(capture_stdout=True, capture_stderr=True, quiet=True)
        except ffmpeg.Error as e:
            print("stderr:", e.stderr.decode("utf8"))
            return None

        return output_video_path

    def create_samples_for_event_group(
        self, video_processor: VideoProcessor, audio_processor: AudioProcessor
    ) -> List[Dict[str, str]]:
        samples_for_event_group = []
        clips = {}
        # Load the video and resize it once. Then, we'll overlay stuff on the video
        try:
            clips["video_clip"] = video_processor.load_and_resize_video(self.video_path)
        except IOError:
            # TODO: Aggregate the errors and return them
            print("Skipping video", self.video_path)
            return []

        # If the video has no audio, skip it
        if clips["video_clip"].audio is None:
            print("Skipping video", self.video_path)
            return []

        # If the audio is too quiet, skip the video
        if audio_processor.calculate_rms(clips["video_clip"].audio) < 1e-9:
            print("Skipping video", self.video_path)
            return []

        compressed_video_path = video_processor.get_compressed_video_path(self.video_path)
        video_processor.save_video(clips["video_clip"], compressed_video_path)

        for event_idx, event in enumerate(self.get_events_for_group()):

            # Check if the event is an overlay event
            if not event.is_overlay():
                continue

            dataset_sample = {
                "raw_video_path": str(self.video_path),
                "compressed_video_path": compressed_video_path,
                "overlayed_event_index": event_idx,
                "events": self.get_event_group_as_dict(),
            }

            # Create a separate video clip just for the event
            clips["event_video"] = video_processor.clip_video(
                video_clip=clips["video_clip"], start=event.start, duration=event.duration
            )
            event_video_path = video_processor.get_event_video_path(self.video_path, event_idx)
            video_processor.save_video(clips["event_video"], event_video_path)
            dataset_sample["event_video_path"] = event_video_path

            # Get random audio class and the audio clip
            audio_class = audio_processor.get_random_audio_class()
            audio_path = audio_processor.get_random_audio_path(audio_class)
            clips["audio_clip"] = audio_processor.create_audio_clip(audio_path)
            dataset_sample["audio_class"] = audio_class
            dataset_sample["audio_path"] = str(audio_path)

            # Prepare audio & overlayed video, and silent video
            clips["composite_audio"] = audio_processor.make_composite_audio(
                video_clip=clips["video_clip"],
                audio_clip=clips["audio_clip"],
                start=event.start,
                duration=event.duration,
            )

            # Prepare video clip with overlayed audio
            clips["video_clip_with_overlayed_audio"] = video_processor.overlay_audio_on_video(
                video_clip=clips["video_clip"], audio_clip=clips["composite_audio"]
            )
            video_with_overlayed_audio_path = video_processor.get_overlayed_video_path(
                self.video_path, audio_class, event_idx
            )
            video_with_overlayed_audio_path = video_processor.save_video(
                video_clip=clips["video_clip_with_overlayed_audio"],
                video_path=video_with_overlayed_audio_path,
            )
            dataset_sample["video_with_overlayed_audio_path"] = video_with_overlayed_audio_path

            # Prepare the silent video clip
            clips["silent_video_clip"] = video_processor.remove_audio_from_video(
                video_clip=clips["video_clip"]
            )
            silent_video_path = video_processor.get_silent_video_path(self.video_path)
            silent_video_path = video_processor.save_video(
                video_clip=clips["silent_video_clip"], video_path=silent_video_path
            )
            dataset_sample["silent_video_path"] = silent_video_path

            # Prepare the composite audio path
            composite_audio_path = audio_processor.get_composite_audio_path(
                video_with_overlayed_audio_path
            )
            composite_audio_path = audio_processor.save_audio(
                audio_clip=clips["composite_audio"], audio_path=composite_audio_path
            )
            dataset_sample["overlayed_audio_path"] = composite_audio_path

            samples_for_event_group.append(dataset_sample)

        # Close the clips
        for clip_name in clips.keys():
            clips[clip_name].close()

        return samples_for_event_group


class EventGroupsForVideo:

    def __init__(self, video_id: str, event_groups: List[EventGroup]):
        self.video_id = video_id
        self.event_groups = event_groups

    def __str__(self):
        header = "*" * 50 + "\n"
        header += f"Video {self.video_id} with {len(self.event_groups)} event groups.\n"
        header += "*" * 50 + "\n"

        if not self.event_groups:
            return header + "No event groups found."

        groups_str = "\n\n".join(
            f"Event Group {i+1}:\n{str(group)}" for i, group in enumerate(self.event_groups)
        )
        return header + groups_str

    def __len__(self):
        return len(self.event_groups)

    def filter_event_groups(self, config: DatasetConfig) -> None:
        self.event_groups = [
            e
            for e in self.event_groups
            # Group must have valid duration
            if e.valid_group_duration(max_duration=config.MAX_SEQUENCE_DURATION)
            # Each event in the group must have valid duration
            and e.valid_event_durations(
                min_duration=config.MINIMUM_EVENT_DURATION,
                max_duration=config.MAXIMUM_EVENT_DURATION,
            )
            # Group must have valid overlap
            and e.valid_overlap(max_overlap=config.MAX_OVERLAP)
            # At least one event has to have a duration above the threshold
            and e.at_least_one_event_above_duration(
                min_duration=config.MIN_EVENT_DURATION_FOR_OVERLAY
            )
            and e.valid_text(
                words_to_filter=config.WORDS_TO_FILTER,
                enforse_unique_events=config.ENFORSE_UNIQUE_EVENTS,
            )
        ]

    def flag_overlay_events(self, config: DatasetConfig) -> None:
        for event_group in self.event_groups:
            event_group.flag_overlay_events(config=config)

    def trim_and_save_videos(
        self,
        dataset_name: str,
        read_data_path: Path,
        write_data_path: Path,
        processed_videos_dir_name: str,
        **kwargs,
    ) -> None:
        for event_group in self.event_groups:
            input_video_path, output_video_path = event_group.get_input_output_video_path(
                dataset_name=dataset_name,
                read_data_path=read_data_path,
                write_data_path=write_data_path,
                video_id=self.video_id,
                processed_videos_dir_name=processed_videos_dir_name,
                **kwargs,
            )
            video_path = event_group.trim_and_save_video(
                input_video_path=input_video_path, output_video_path=output_video_path
            )
            if not video_path:
                continue
            event_group.adjust_event_times()
            event_group.set_video_path(video_path=video_path)

    def add_video_id_to_dataset_samples(
        self, samples: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        for sample in samples:
            sample["video_id"] = self.video_id
        return samples

    def create_samples_for_video(
        self,
        video_processor: VideoProcessor,
        audio_processor: AudioProcessor,
    ) -> List[Dict[str, str]]:
        if not self.event_groups:
            return []

        samples_for_video = []
        for event_group in tqdm(
            self.event_groups, leave=True, desc=f"Processing video {self.video_id}"
        ):
            if not event_group.has_video_path():
                continue
            # Create the samples for the event group
            event_group_samples = event_group.create_samples_for_event_group(
                video_processor, audio_processor
            )

            # Add the video ID to each sample
            event_group_samples = self.add_video_id_to_dataset_samples(event_group_samples)

            # Add the samples to the list of samples for the video
            samples_for_video.extend(event_group_samples)

        return samples_for_video


class EventsForVideo:

    def __init__(self, video_id: str, events: Optional[List[Event]] = None):
        self.video_id = video_id
        self.events = events if events else []

    def __str__(self):
        string = "*" * 50 + "\n"
        string += f"Video {self.video_id} with {len(self.events)} events.\n"
        string += "*" * 50 + "\n"
        string += "\n".join(str(e) for e in self.events)
        return string

    def __getitem__(self, idx: int):
        return self.events[idx]

    def add_event(self, event: Event):
        self.events.append(event)

    def sort_event_list_by_start_time(self) -> "EventsForVideo":
        return EventsForVideo(self.video_id, sorted(self.events, key=lambda e: e.start))

    def clean_zero_duration_events(self) -> "EventsForVideo":
        return EventsForVideo(self.video_id, events=[e for e in self.events if e.duration != 0])

    def merge_consequtive_events_same_action(self) -> "EventsForVideo":
        if not self.events:
            return EventsForVideo(self.video_id, [])
        merged_events = []
        current_event = self.events[0]

        for i in range(1, len(self.events)):
            next_event = self.events[i]

            if (
                current_event.action == next_event.action
                or current_event.narration == next_event.narration
            ):
                current_event = current_event.merge_with(next_event)
            else:
                merged_events.append(current_event)
                current_event = next_event

        # Add the last event
        merged_events.append(current_event)
        return EventsForVideo(video_id=self.video_id, events=merged_events)

    def get_events_groups(self) -> EventGroupsForVideo:
        return EventGroupsForVideo(
            video_id=self.video_id,
            event_groups=[EventGroup(self.events[i : i + 4]) for i in range(len(self.events) - 3)],
        )


class VideosForParticipant:

    def __init__(self, video_id2events: Dict[str, EventsForVideo], participant_id: str):
        # For each participant, we have a list of videos that we can store, and for each video we have a list of events
        self.video_id2events = video_id2events
        self.participant_id = participant_id
        self.event_groups_created = False

    def get_video_ids(self) -> List[str]:
        return list(self.video_id2events.keys())

    def get_video_data(self, video_id: str) -> EventsForVideo:
        return self.video_id2events[video_id]

    def __str__(self):
        header = f"Participant {self.participant_id} with {len(self.video_id2events)} videos.\n"

        video_strings = []
        for video_id, content in self.video_id2events.items():
            if isinstance(content, EventGroupsForVideo):  # For EventGroup lists
                video_str = "*" * 50 + "\n"
                video_str += f"Video {video_id} with {len(content)} event groups.\n"
                # Separate each group by a * separator
                separator = "\n" + "*" * 50 + "\n"
                video_str += separator.join(str(group) for group in content.event_groups)
            else:  # For EventsForVideo objects
                video_str = str(content)
            video_strings.append(video_str)

        return header + "\n".join(video_strings)

    def num_valid_event_groups(self) -> int:
        assert self.event_groups_created, "Event groups not yet created."
        return sum(len(v) for v in self.video_id2events.values())

    @classmethod
    def from_csv_file(cls, file_path: Path, participant_id: str) -> "VideosForParticipant":
        video_id2events = {}
        with open(file_path, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row["participant_id"] != participant_id:
                    continue

                event = Event.from_epic_kitchens_row(row)
                if not event:
                    continue

                video_id = row["video_id"]
                if event.video_id not in video_id2events:
                    video_id2events[video_id] = EventsForVideo(video_id=video_id)
                video_id2events[video_id].add_event(event)

        return cls(video_id2events, participant_id=participant_id)

    def sort_event_list_by_start_time(self) -> None:
        self.video_id2events = {
            k: v.sort_event_list_by_start_time() for k, v in self.video_id2events.items()
        }

    def merge_consequtive_events_same_action(self) -> None:
        self.video_id2events = {
            k: v.merge_consequtive_events_same_action() for k, v in self.video_id2events.items()
        }

    def convert_events_to_event_groups(self) -> None:
        self.event_groups_created = True
        self.video_id2events = {k: v.get_events_groups() for k, v in self.video_id2events.items()}

    def filter_event_groups(self, config: DatasetConfig) -> None:
        for video_event_group in self.video_id2events.values():
            video_event_group.filter_event_groups(config=config)

    def flag_overlay_events(self, config: DatasetConfig) -> None:
        for video_event_group in self.video_id2events.values():
            video_event_group.flag_overlay_events(config=config)

    def trim_and_save_videos(
        self,
        config: DatasetConfig,
        read_data_path: Path,
        write_data_path: Path,
    ) -> None:
        for _, event_groups in self.video_id2events.items():
            event_groups.trim_and_save_videos(
                dataset_name="epic",
                read_data_path=read_data_path,
                write_data_path=write_data_path,
                processed_videos_dir_name=config.PROCESSED_VIDEOS_DIR_NAME,
                participant_id=self.participant_id,
            )

    def add_participant_id_to_dataset_samples(
        self, samples: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        for sample in samples:
            sample["participant_id"] = self.participant_id
        return samples

    def create_samples_for_participant(
        self, video_processor: VideoProcessor, audio_processor: AudioProcessor
    ):
        samples_for_participant = []
        for event_groups in self.video_id2events.values():
            # Create the samples for the video
            video_samples = event_groups.create_samples_for_video(video_processor, audio_processor)

            # Add the participant ID to each sample
            video_samples = self.add_participant_id_to_dataset_samples(video_samples)

            # Add the samples to the list of samples for the participant
            samples_for_participant.extend(video_samples)

        return samples_for_participant


class Ego4dVideoProcessor:

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

    def set_video2events(self, json_file_path: Path, videos_path: Path):
        self.video2events: Dict[str, EventGroupsForVideo] = {}
        json_file = json.load(open(json_file_path))
        video_ids = list(json_file.keys())
        for video_id in tqdm(video_ids):

            video_path = videos_path / f"{video_id}.mp4"
            if not video_path.exists():
                continue

            if json_file[video_id].get("status") != "complete":
                continue

            events = []
            narrations = json_file[video_id]["narration_pass_2"]["narrations"]
            for idx in range(len(narrations) - 1):
                # Get current and next event data
                cur_event_data = narrations[idx]
                next_event_data = narrations[idx + 1]

                cur_start_time, cur_end_time = (
                    cur_event_data["timestamp_sec"],
                    next_event_data["timestamp_sec"],
                )

                if cur_end_time < cur_start_time:
                    continue

                narration = re.sub(r'[^\w\s]', '', cur_event_data["narration_text"])

                event = Event(
                    start=cur_start_time,
                    end=cur_end_time,
                    narration=narration,
                    action=narration,
                    video_id=video_id,
                )
                events.append(event)

            events_for_video = EventsForVideo(video_id=video_id, events=events)
            events_for_video = events_for_video.clean_zero_duration_events()
            events_for_video = events_for_video.sort_event_list_by_start_time()
            events_for_video = events_for_video.merge_consequtive_events_same_action()
            event_groups = events_for_video.get_events_groups()
            event_groups.filter_event_groups(config=self.config)
            event_groups.flag_overlay_events(config=self.config)
            self.video2events[video_id] = event_groups

    def get_video_ids(self) -> List[str]:
        return list(self.video2events.keys())

    def __call__(self, video_id: str):
        video_directory_path = (
            self.write_data_path / self.config.PROCESSED_VIDEOS_DIR_NAME / f"{video_id}_processed"
        )
        if not video_directory_path.exists():
            # If the video_directory_path starts with /readonly, remove it
            if str(video_directory_path).startswith("/readonly"):
                video_directory_path = video_directory_path.relative_to("/readonly")
            video_directory_path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"Folder {video_directory_path} already exists! Skipping {video_id}")
            return

        event_groups_for_video = self.video2events[video_id]

        event_groups_for_video.trim_and_save_videos(
            config=self.config,
            read_data_path=self.read_data_path,
            write_data_path=self.write_data_path,
            dataset_name="ego4d",
            processed_videos_dir_name=self.config.PROCESSED_VIDEOS_DIR_NAME,
        )
        samples_for_video = event_groups_for_video.create_samples_for_video(
            video_processor=self.video_processor, audio_processor=self.audio_processor
        )

        return samples_for_video
