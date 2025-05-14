import re
import time
from pathlib import Path
from typing import Dict

import cv2
import google.generativeai as generativeai
from google import genai
from google.api_core.exceptions import (
    InternalServerError,
    ResourceExhausted,
    ServiceUnavailable,
)
from google.genai import types
from google.generativeai.types.file_types import File as GenAiFile
from PIL import Image

from models import BaseModel
from utils.constants import GOOGLE_API_KEY
from utils.prompts import prompt_factory


class GeminiModel(BaseModel):
    """Handles video file processing and GenAI interactions."""

    def __init__(self, **kwargs):
        model_name = kwargs.get("model_name", "models/gemini-2.0-flash-lite")
        generativeai.configure(api_key=GOOGLE_API_KEY)
        self.model = generativeai.GenerativeModel(
            model_name=model_name, generation_config={"temperature": 0.0}
        )
        self.google_id_mapping = kwargs.pop("google_id_mapping", None)
        assert self.google_id_mapping is not None, "Google ID mapping is required for GeminiModel"

    def get_uploaded_file(self, file_path: Path) -> GenAiFile:
        """Get the GenAI file using the pre-uploaded file ID."""
        # Get file_path relative to the data directory
        file_id = self.google_id_mapping[str(file_path)]
        genai_file = None
        for _ in range(10):
            try:
                genai_file = generativeai.get_file(file_id)
            except (ServiceUnavailable, InternalServerError):
                print("Service unavailable. Retrying after 1 minute...")
                time.sleep(60)
        if genai_file is None:
            raise Exception("Failed to get file after 10 attempts")

        if genai_file.state.name == "FAILED":
            raise ValueError(f"File is in FAILED state: {file_path}")
        return genai_file

    def prepare_model_input(self, sample: Dict, prompt_suffix: str, **kwargs) -> Dict:
        """Prepare the input for the model based on prompt type."""
        sound_name = sample.get("sound_name", None)
        raw_choices = sample.get("raw_choices", None)
        question = sample.get("question", None)

        model_input = {
            "prompt": prompt_factory[sample["prompt_type"]](
                sound_name=sound_name,
                raw_choices=raw_choices,
                suffix=prompt_suffix,
                question=question,
            )
        }

        # If text_only prompt, that means there is no other modality
        if sample["prompt_type"] == "text_only":
            return model_input

        # Get file_path relative to the data directory
        if isinstance(sample["file_path"], dict):
            model_input["file_path"] = {
                k: Path(v)
                for k, v in sample["file_path"].items()
            }
        else:
            model_input["file_path"] = Path(sample["file_path"])

        return model_input

    def inference(
        self, model_input: Dict[str, str | Dict[str, str]], prompt_type: str, **kwargs
    ) -> Dict:
        """Process the model input and handle retries on failure."""
        # Replace the file path with the GenAI file
        if "file_path" in model_input:
            if isinstance(model_input["file_path"], str) or isinstance(
                model_input["file_path"], Path
            ):
                model_input["genai_file"] = self.get_uploaded_file(model_input["file_path"])
            # In this case we assert whether the prompt_type is multimodal, and we get only the video part
            elif isinstance(model_input["file_path"], dict):
                assert prompt_type == "multimodal", "Prompt type must be multimodal!"
                model_input["genai_file"] = self.get_uploaded_file(
                    model_input["file_path"]["video"]
                )

        content = (
            [model_input["genai_file"], model_input["prompt"]]
            if "genai_file" in model_input
            else model_input["prompt"]
        )

        # Try to obtain response with retries
        for attempt in range(10):
            try:
                response = self.model.generate_content(content, request_options={"timeout": 600})
                break  # Exit loop immediately on success
            except (ResourceExhausted, InternalServerError):
                if attempt == 9:  # Last attempt
                    raise Exception("Failed to generate content after 10 attempts")
                print(f"Attempt {attempt + 1} failed. Retrying after 1 minute...")
                time.sleep(60)

        try:
            response_text = response.text.strip()
        except ValueError:
            return {"response_text": "No answer found!"}

        # For temporal video task, we need to extract a sequence of choices
        if prompt_type == "temporal_video":
            matches = re.findall(r"\(([A-E])\)", response_text)
            final_response = [f"({l})" for l in matches] if matches else "No valid sequence found!"
            return {"response_text": final_response}

        match = re.search(r"\((\w)\)", response_text)
        return {
            "response_text": f"({match.group(1)})" if match else "No answer found!",
            "llm_output": response_text,
        }


class GeminiPipeline(BaseModel):
    def __init__(self, **kwargs):
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model_name = "gemini-1.5-flash-latest"

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
            raise ValueError(f"Invalid format in LLM output: {llm_output}")

    @staticmethod
    def extract_frames(video_path, timestamp_start, timestamp_end, fps=10):
        """
        Extracts frames as PIL images at the given FPS between timestamp_start and timestamp_end.

        Args:
            video_path (str): Path to the video file.
            timestamp_start (float): Start time in seconds.
            timestamp_end (float): End time in seconds.
            fps (int): Frames per second to extract.

        Returns:
            List[PIL.Image]: List of extracted frames as PIL images.
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error: Could not open video file.")

        # Get video FPS and frame count
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        # Compute the frame numbers for extraction
        frame_start = int(timestamp_start * video_fps)
        frame_end = int(timestamp_end * video_fps)
        frame_interval = int(video_fps / fps)  # Interval between frames

        # Read frames
        frames = []
        for frame_number in range(frame_start, frame_end, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                break  # Stop if the video ends

            # Convert frame to PIL image (OpenCV uses BGR, so convert to RGB)
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(pil_image)

        # Release video capture
        cap.release()

        # print(len(frames), "frames extracted from", timestamp_start, "to", timestamp_end, "seconds.")
        return frames

    @staticmethod
    def get_audio_prompt(sound_name: str):
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
        assert sample["sound_name"] is not None, "Sound name is required for GeminiPipeline!"

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

    def prompt_to_extract_timestamps(self, model_input):
        audio_path = model_input["file_path"]["audio"]
        audio_file = self.client.files.upload(file=audio_path)

        audio_response = self.client.models.generate_content(
            model=self.model_name,
            config=types.GenerateContentConfig(
                max_output_tokens=20,
                temperature=0.0,
            ),
            contents=[model_input["audio_prompt"], audio_file],
        )

        return audio_response.text

    def inference(
        self, model_input: Dict[str, str | Dict[str, str]], prompt_type: str, **kwargs
    ) -> Dict:
        """Process the model input and handle retries on failure."""
        assert prompt_type == "multimodal" or prompt_type == "pipeline_event_classification"

        if prompt_type == "multimodal":
            audio_response = self.prompt_to_extract_timestamps(model_input)
            if "none" in audio_response.lower():
                # NOTE: we assume that None of the above is always option (E)
                return {"response_text": "(E)"}
            timestamp_start, timestamp_end = self.extract_timestamps(audio_response)
            file_path = model_input["file_path"]["video"]

        elif prompt_type == "pipeline_event_classification":
            timestamp_start, timestamp_end = model_input["ground_truth_timestamps"]
            file_path = model_input["file_path"]

        frames = self.extract_frames(file_path, timestamp_start, timestamp_end)

        video_response = self.client.models.generate_content(
            model=self.model_name,
            config=types.GenerateContentConfig(
                max_output_tokens=5,
                temperature=0.0,
            ),
            contents=[f"{model_input['prompt']}", *frames],
        )

        match = re.search(r"\((\w)\)", video_response.text)
        response_text = f"({match.group(1)})" if match else "No answer found!"

        return {
            "response_text": response_text,
            "predicted_timestamps": [timestamp_start, timestamp_end],
            "llm_output": video_response.text,
        }
