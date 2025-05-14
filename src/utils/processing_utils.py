import csv
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from moviepy import (
    AudioFileClip,
    CompositeAudioClip,
    VideoFileClip,
    concatenate_audioclips,
)
from moviepy.audio.fx import AudioFadeIn, AudioFadeOut

from utils.config import DatasetConfig
from utils.constants import BASE_DATETIME

# Suppress specific UserWarnings from moviepy
warnings.filterwarnings("ignore", category=UserWarning, module="moviepy")


class VideoProcessor:

    def __init__(self, config: DatasetConfig):
        self.config = config

    def load_and_resize_video(self, video_path: Path):
        video = VideoFileClip(video_path).resized(self.config.VIDEO_RESIZE_PERCENTAGE)
        return video

    def overlay_audio_on_video(
        self, video_clip: VideoFileClip, audio_clip: AudioFileClip
    ) -> VideoFileClip:
        return video_clip.with_audio(audio_clip)

    def get_compressed_video_path(self, video_path: Path | str, suffix: str = "compressed") -> str:
        if isinstance(video_path, str):
            video_path = Path(video_path)
        return str(video_path.with_stem(f"{video_path.stem}+{suffix}"))

    def get_event_video_path(self, video_path: Path | str, event_idx: int) -> str:
        if isinstance(video_path, str):
            video_path = Path(video_path)
        event_video_stem = f"{video_path.stem}+segment_{event_idx}"
        event_video_path = video_path.with_stem(event_video_stem)
        return str(event_video_path)

    def get_silent_video_path(self, video_path: Path | str) -> str:
        if isinstance(video_path, str):
            video_path = Path(video_path)
        return str(video_path.with_stem(f"{video_path.stem}+silent"))

    def get_overlayed_video_path(
        self, video_path: Path | str, audio_class: str, event_idx: int
    ) -> str:
        if isinstance(video_path, str):
            video_path = Path(video_path)
        video_with_overlayed_audio_stem = f"{video_path.stem}+{audio_class}_{event_idx}"
        video_with_overlayed_audio_path = video_path.with_stem(video_with_overlayed_audio_stem)
        return str(video_with_overlayed_audio_path)

    def save_video(self, video_clip: VideoFileClip, video_path: Path | str) -> Optional[str]:
        if isinstance(video_path, str):
            video_path = Path(video_path)
        try:
            if video_path.exists():
                return str(video_path)

            video_clip.write_videofile(video_path, logger=None)
            return str(video_path)
        except OSError:
            return None

    def remove_audio_from_video(self, video_clip: VideoFileClip) -> VideoFileClip:
        return video_clip.without_audio()

    def clip_video(self, video_clip: VideoFileClip, start: datetime, duration: float):
        start_in_seconds = (start - BASE_DATETIME).total_seconds()
        end_in_seconds = start_in_seconds + duration
        return video_clip.subclipped(start_time=start_in_seconds, end_time=end_in_seconds)


class AudioProcessor:

    def __init__(self, config: DatasetConfig, esc_path: Path):
        self.config = config
        self.class2path_sounds = self.get_audio_class_to_paths_mapping(esc_path)

    def get_audio_class_to_paths_mapping(self, esc_path: Path) -> dict[str, List[Path]]:
        class2path = {}
        with open(esc_path / "meta" / "esc50.csv", mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                category = row["category"]
                if category not in class2path:
                    class2path[category] = []
                class2path[category].append(esc_path / "audio" / row["filename"])

        return class2path

    def get_composite_audio_path(self, video_path: str | Path) -> str:
        if isinstance(video_path, str):
            video_path = Path(video_path)
        return str(video_path.with_suffix(".mp3"))

    def get_random_audio_class(self):
        return random.choice(self.config.SOUND_CLASSES)

    def get_random_audio_path(self, audio_class: str):
        return random.choice(self.class2path_sounds[audio_class])

    def create_audio_clip(self, audio_path: str):
        return AudioFileClip(audio_path)

    @staticmethod
    def calculate_rms(audio_clip: AudioFileClip) -> float:
        samples = audio_clip.to_soundarray(fps=44100)
        rms = np.sqrt(np.mean(samples**2))
        return rms

    def scale_audio_volumes(
        self,
        video_audio: AudioFileClip,
        overlay_audio: AudioFileClip,
        target_ratio: float = 1.0,  # Desired ratio between video and overlay audio
    ) -> tuple[AudioFileClip, AudioFileClip]:
        """
        Scale both audio tracks to maintain a desired ratio between their volumes.
        Returns the scaled video audio and overlay audio.
        """
        video_rms = self.calculate_rms(video_audio)
        overlay_rms = self.calculate_rms(overlay_audio)
        current_ratio = video_rms / overlay_rms

        # If current_ratio > target_ratio, overlay is too quiet
        # If current_ratio < target_ratio, overlay is too loud
        adjustment_factor = np.sqrt(target_ratio / current_ratio)

        # Scale both audio tracks inversely to maintain overall volume
        video_scale = 1 / adjustment_factor
        overlay_scale = adjustment_factor

        # Apply scaling coefficient to adjust the balance
        if self.config.AUDIO_SCALE_COEFFICIENT != 0:
            video_scale *= 2 - self.config.AUDIO_SCALE_COEFFICIENT
            overlay_scale *= self.config.AUDIO_SCALE_COEFFICIENT

        return (
            video_audio.with_volume_scaled(video_scale),
            overlay_audio.with_volume_scaled(overlay_scale),
        )

    def extend_audio_duration(self, audio_clip: AudioFileClip, duration: float):
        if audio_clip.duration < duration:
            num_repeats = int(duration // audio_clip.duration) + 1
            extended_audio = concatenate_audioclips([audio_clip] * num_repeats)
            return extended_audio.subclipped(0, duration)
        else:
            return audio_clip.subclipped(0, duration)

    def fade_in_and_out_audio(self, audio_clip: AudioFileClip):
        return audio_clip.with_effects(
            [
                AudioFadeIn(self.config.FADE_IN_OUT_DURATION),
                AudioFadeOut(self.config.FADE_IN_OUT_DURATION),
            ]
        )

    def make_composite_audio(
        self,
        video_clip: VideoFileClip,
        audio_clip: AudioFileClip,
        start: datetime,
        duration: float,
    ) -> AudioFileClip:
        audio_clip = self.extend_audio_duration(audio_clip=audio_clip, duration=duration)

        if self.config.AUDIO_SCALE_COEFFICIENT != 0:
            # Scale both audio tracks
            scaled_video_audio, scaled_overlay_audio = self.scale_audio_volumes(
                video_audio=video_clip.audio, overlay_audio=audio_clip, target_ratio=1.0
            )
            # Replace the audio tracks with scaled versions
            video_clip = video_clip.with_audio(scaled_video_audio)
            audio_clip = scaled_overlay_audio
        # Overlay the audio on the video: set start and end times on the merged audio file based on the event data. The
        # event data contains the start and end times of the event where we overlay the audio.

        if self.config.FADE_IN_OUT_DURATION:
            audio_clip = self.fade_in_and_out_audio(audio_clip=audio_clip)

        # Offset the audio start for a fixed offset as naturally the audio starts a bit earlier
        start_in_seconds = (start - BASE_DATETIME).total_seconds()
        start_in_seconds += self.config.AUDIO_START_OFFSET

        # End is the start + the duration of the event
        end_in_seconds = start_in_seconds + duration

        audio_clip = audio_clip.with_start(start_in_seconds).with_end(end_in_seconds)
        composite_audio = CompositeAudioClip([video_clip.audio, audio_clip])
        return composite_audio

    def save_audio(self, audio_clip: AudioFileClip, audio_path: Path | str) -> Optional[str]:
        if isinstance(audio_path, str):
            audio_path = Path(audio_path)
        try:
            if audio_path.exists():
                return None

            # Save the composite audio as a separate file
            audio_clip.write_audiofile(audio_path, logger=None)
            return str(audio_path)
        except OSError:
            return None
