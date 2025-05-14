import warnings
from pathlib import Path
from typing import Dict

from utils.prompts import prompt_factory

warnings.filterwarnings("ignore")
from models import BaseModel

# (1) Place the videollama2 model somewhere in the repository (e.g., in external, and import it)
# from external.video_llama_audio.videollama2 import mm_infer, model_init
# (2) Then remove the dummy methods


def model_init(*args, **kwargs):
    pass


def mm_infer(*args, **kwargs):
    pass


class VideoLlamaModel(BaseModel):

    def __init__(self, **kwargs):
        model_path = "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"
        self.model, self.processor, self.tokenizer = model_init(model_path)

    def inference(
        self,
        model_input: Dict[str, str | Path | Dict[str, Path]],
        prompt_type: str,
        **kwargs,
    ) -> Dict:
        prompt_category = model_input["prompt_category"]
        if prompt_category == "audio":
            self.model.vision_tower = None
            preprocess, file_path, modal, va = (
                self.processor["audio"],
                model_input["file_path_audio"],
                "audio",
                None,
            )
        elif prompt_category == "video":
            self.model.audio_tower = None
            preprocess, file_path, modal, va = (
                self.processor["video"],
                model_input["file_path_video"],
                "video",
                False,
            )
        elif prompt_category == "multimodal":
            preprocess, file_path, modal, va = (
                self.processor["video"],
                model_input["file_path_video"],
                "video",
                True,
            )
        else:
            raise NotImplementedError(f"Unsupported prompt type: {prompt_type}")

        audio_video_tensor = preprocess(str(file_path), **({"va": va} if va is not None else {}))

        response_text = mm_infer(
            audio_video_tensor,
            model_input["prompt"],
            model=self.model,
            tokenizer=self.tokenizer,
            do_sample=False,
            modal=modal,
        )

        matched_letter = self.extract_letter(response_text)

        return {
            "response_text": (f"({matched_letter})" if matched_letter else "No answer found!"),
            "llm_output": response_text,
        }

    def prepare_model_input(self, sample: Dict, prompt_suffix: str, **kwargs) -> Dict:
        sound_name = sample.get("sound_name", None)
        raw_choices = sample.get("raw_choices", None)
        question = sample.get("question", None)

        prompt_type = sample["prompt_type"]
        file_path = sample["file_path"]

        if prompt_type == "multimodal":
            video_file_path, audio_file_path = file_path.get("video"), file_path.get("audio")
            prompt_category = "multimodal"
        elif prompt_type in [
            "audio",
            "overlayed_full_audio_classification",
            "simple_audio_classification",
        ]:
            video_file_path, audio_file_path = None, file_path
            prompt_category = "audio"
        elif prompt_type in ["temporal_video", "silent_video", "video_segment"]:
            video_file_path, audio_file_path = file_path, None
            prompt_category = "video"
        else:
            raise NotImplementedError(f"Unsupported prompt type: {prompt_type}")

        model_input = {
            "prompt": prompt_factory[sample["prompt_type"]](
                sound_name=sound_name,
                raw_choices=raw_choices,
                suffix=prompt_suffix,
                question=question,
            ),
            "file_path_video": video_file_path,
            "file_path_audio": audio_file_path,
            "prompt_category": prompt_category,
        }

        return model_input
