import base64
import re
from pathlib import Path
from typing import Dict, Union

import cv2
from openai import OpenAI

from models import BaseModel
from utils.prompts import prompt_factory


class GPTAudioModel:
    """Handles audio file processing"""

    def __init__(self, **kwargs):
        self.model = OpenAI()

    @staticmethod
    def encode_audio_to_base64(audio_path: str):
        # Open the audio file in binary mode and read its contents
        with open(audio_path, "rb") as audio_file:
            audio_data = audio_file.read()

        # Encode the binary data into base64
        return base64.b64encode(audio_data).decode(
            "utf-8"
        )  # Decode to 'utf-8' for a string representation

    @staticmethod
    def extract_timestamps(llm_output):
        """
        Extracts timestamps from the LLM output, handling:
        - MM:SS format (e.g., [00:05, 00:08])
        - HH:MM:SS format (e.g., [00:00:00, 00:00:03])
        - Floating-point seconds (e.g., [0.0, 5.0])

        Args:
            llm_output (str): The LLM response.

        Returns:
            tuple: (timestamp_start, timestamp_end) as (float, float)
        """
        # Normalize input by stripping square brackets if present
        llm_output = llm_output.strip("[]")

        # Regex to match two timestamps in different formats
        match = re.match(r"\s*([\d:.]+)\s*,\s*([\d:.]+)\s*", llm_output)

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
            print("Invalid format in LLM output: {llm_output}")
            return None, None
            # raise ValueError(f"Invalid format in LLM output: {llm_output}")

    def prompt_to_extract_timestamps(self, prompt: str, audio_path: str):
        encoded_string = self.encode_audio_to_base64(audio_path)
        completion = self.model.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "mp3"},
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in audio event detection and classification.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": encoded_string, "format": "mp3"},
                        },
                    ],
                },
            ],
        )

        llm_output = completion.choices[0].message.audio.transcript
        return llm_output


class GPTVideoModel:

    def __init__(self, **kwargs):
        self.model = OpenAI()

    @staticmethod
    def extract_frames(video_path, timestamp_start, timestamp_end, sample_fps=10):
        """
        Args:
            video_path (str): Path to the video file.
            timestamp_start (float): The start time in seconds.
            timestamp_end (float): The end time in seconds.
            sample_fps (int): Number of frames per second to extract.
        """
        video = cv2.VideoCapture(video_path)

        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)  # Original frames per second
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Convert timestamps to frame indices
        start_frame = int(timestamp_start * fps)
        end_frame = int(timestamp_end * fps)

        # Ensure timestamps are within video bounds
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(0, min(end_frame, total_frames - 1))

        # Determine frame interval based on the desired sample FPS
        frame_interval = max(1, int(fps / sample_fps))

        # Move to the start frame
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        base64Frames = []
        frame_count = start_frame

        while video.isOpened() and frame_count <= end_frame:
            success, frame = video.read()
            if not success:
                break

            # Process frames at the specified sample_fps interval
            if (frame_count - start_frame) % frame_interval == 0:
                _, buffer = cv2.imencode(".jpg", frame)
                base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

            frame_count += 1

        video.release()

        return base64Frames

    def prompt_to_answer_multichoice(
        self, prompt, video_path, timestamp_start, timestamp_end
    ):
        frames = self.extract_frames(video_path, timestamp_start, timestamp_end)
        PROMPT_MESSAGES = [
            {
                "role": "user",
                "content": [
                    f"{prompt}",
                    *map(lambda x: {"image": x, "resize": 768}, frames),
                ],
            },
        ]
        params = {
            "model": "gpt-4o",
            "messages": PROMPT_MESSAGES,
            "max_tokens": 5,
        }

        result = self.model.chat.completions.create(**params)
        response_text = result.choices[0].message.content

        return response_text


class GPTAudioVideoModel(BaseModel):

    def __init__(self, **kwargs):
        self.audio_model = GPTAudioModel()
        self.video_model = GPTVideoModel()

    def get_audio_prompt(self, sound_name: str):
        return (
            "Please listen to the audio. "
            f"There are several kitchen-environment sounds, but one ({sound_name}) is different or out of place. "
            f"Please identify the timestamp (start and end) when {sound_name} occurs. "
            "Only output [timestamp_start, timestamp_end] in MM:SS format."
            "Note that the sound might not be present at all, in which case only output 'None'."
        )

    def prepare_model_input(self, sample: Dict, prompt_suffix: str, **kwargs) -> Dict:
        assert (
            sample["prompt_type"] == "multimodal"
            or sample["prompt_type"] == "pipeline_event_classification"
        )
        assert (
            sample["sound_name"] is not None
        ), "Sound name is required for GPTAudioVideoModel!"
        # Use the appropriate prompt generator for all prompt types
        model_input = {
            "prompt": prompt_factory["pipeline_video"](
                sound_name=sample["sound_name"],
                raw_choices=sample["raw_choices"],
                suffix=prompt_suffix,
            ),
            "audio_prompt": self.get_audio_prompt(sample["sound_name"]),
            "file_path": sample["file_path"],
            "ground_truth_timestamps": sample["ground_truth_timestamps"],
        }

        return model_input

    def inference(
        self,
        model_input: Dict[str, Union[str, Dict[str, Path]]],
        prompt_type: str,
        **kwargs,
    ) -> Dict:
        assert (
            prompt_type == "multimodal"
            or prompt_type == "pipeline_event_classification"
        )

        if prompt_type == "multimodal":
            audio_response = self.audio_model.prompt_to_extract_timestamps(
                prompt=model_input["audio_prompt"],
                audio_path=model_input["file_path"]["audio"],
            )

            if "none" in audio_response.lower():
                # NOTE: we assume that None of the above is always option (E)
                return {"response_text": "(E)"}
            timestamp_start, timestamp_end = self.audio_model.extract_timestamps(
                audio_response
            )
            file_path = model_input["file_path"]["video"]

            if not timestamp_start:
                return {"response_text": "Error", "llm_output": audio_response}

        elif prompt_type == "pipeline_event_classification":
            timestamp_start, timestamp_end = model_input["ground_truth_timestamps"]
            file_path = model_input["file_path"]

        video_response = self.video_model.prompt_to_answer_multichoice(
            prompt=model_input["prompt"],
            video_path=file_path,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
        )

        match = re.search(r"\((\w)\)", video_response)
        response_text = f"({match.group(1)})" if match else "No answer found!"

        return {
            "response_text": response_text,
            "predicted_timestamps": [timestamp_start, timestamp_end],
            "llm_output": video_response,
        }
