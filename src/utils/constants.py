import os
from datetime import datetime

model_choices = [
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
    "models/gemini-2.0-flash-lite",
    "gemini-2.0-flash-001" "random",
]

QUESTION_TYPES = ['joint', 'regular', 'none_of_the_above_no_sound', 'none_of_the_above_incorrect_audio']
PROMPT_TYPES = ['multimodal', 'silent_video', 'audio', 'text_only']
TASKS = ['multimodal', 'temporal_video', 'overlayed_full_audio_classification', "video_segment"]

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

BASE_DATETIME = datetime(1900, 1, 1)
