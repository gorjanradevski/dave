import random
import re
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseModel(ABC):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    # model_input: Dict[str, str | Path | Dict[str, Path]]
    def inference(self, model_input: Dict, prompt_type: str, **kwargs) -> Dict:
        pass

    @abstractmethod
    def prepare_model_input(
        self, sample: Dict, prompt_suffix: str, **kwargs
    ) -> Dict[Any, Any]:
        pass

    @staticmethod
    def extract_letter(output: str) -> str:
        """Extracts a single-letter response from the model output."""
        match_with_brackets = re.search(r"\((\w)\)", output)
        if match_with_brackets:
            return match_with_brackets.group(1)

        match_no_brackets = re.search(r"\b[A-Z]\b", output)

        return match_no_brackets.group(0) if match_no_brackets else None


class RandomModel(BaseModel):

    def __init__(self, **kwargs):
        pass

    # model_input: Dict[str, str | Path | Dict[str, Path]]
    def inference(self, model_input: Dict, prompt_type: str, **kwargs) -> Dict:
        """Generate a random response for a model."""
        if prompt_type == "temporal_video":
            letters = list("ABCD")
            random.shuffle(letters)
            return {"response_text": [f"({l})" for l in letters]}
        elif prompt_type in ["multimodal", "silent_video", "audio", "text_only"]:
            return {"response_text": random.choice(["(A)", "(B)", "(C)", "(D)", "(E)"])}
        elif prompt_type in [
            "simple_audio_classification",
            "overlayed_full_audio_classification",
            "video_segment",
        ]:
            return {"response_text": random.choice(["(A)", "(B)", "(C)", "(D)"])}
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

    def prepare_model_input(self, sample: Dict, prompt_suffix: str, **kwargs):
        return {"prompt": {}}


def model_factory(model_name: str) -> BaseModel:
    if model_name == "random":
        return RandomModel
    if model_name == "gemini-pipeline":
        from models.gemini_model import GeminiPipeline

        return GeminiPipeline
    elif "gemini" in model_name:
        from models.gemini_model import GeminiModel

        return GeminiModel
    elif model_name == "openai":
        from models.openai_model import GPTAudioVideoModel

        return GPTAudioVideoModel
    elif model_name == "pandagpt":
        from models.pandagpt_model import PandaGPTModel

        return PandaGPTModel
    elif model_name == "videollama":
        from models.videollama_model import VideoLlamaModel

        return VideoLlamaModel

    else:
        raise ValueError(f"Unknown model name: {model_name}")
