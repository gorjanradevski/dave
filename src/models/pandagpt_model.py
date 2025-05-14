from pathlib import Path
from typing import Dict

import torch

from models import BaseModel
from utils.prompts import prompt_factory

# (1) Place the openllama model somewhere within the repository and import it
# from external.pandagpt.code.model.openllama import OpenLLAMAPEFTModel
# (2) Then remove the dummy model;


class OpenLLAMAPEFTModel:

    def __init__(self):
        pass

    def __call__(self):
        pass


# (3) Create the checkpoints path where the downloaded checkpoints are placed
CHECKPOINTS_PATH = Path("path_for_the_pretrained_checkpoints")


class PandaGPTModel(BaseModel):

    def __init__(self, **kwargs):
        args = {
            "model": "openllama_peft",
            "imagebind_ckpt_path": CHECKPOINTS_PATH / "imagebind_ckpt",
            "vicuna_ckpt_path": CHECKPOINTS_PATH / "vicuna_ckpt" / "vicuna-13b-v1.5",
            "delta_ckpt_path": CHECKPOINTS_PATH / "pandagpt_ckpt" / "13b" / "pytorch_model.pt",
            "stage": 2,
            "max_tgt_len": 128,
            "lora_r": 32,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
        }
        self.model = OpenLLAMAPEFTModel(**args)
        delta_ckpt = torch.load(args["delta_ckpt_path"], map_location=torch.device("cpu"))
        self.model.load_state_dict(delta_ckpt, strict=False)
        self.model = self.model.eval()
        self.model = self.model.half().cuda()

    def inference(
        self,
        model_input: Dict[str, str | Path | Dict[str, Path]],
        prompt_type: str,
        **kwargs,
    ) -> Dict:

        response_text = self.model.generate(
            {
                "prompt": model_input["prompt"],
                "image_paths": [],
                "audio_paths": (
                    [model_input["file_path_audio"]] if model_input["file_path_audio"] else []
                ),
                "video_paths": (
                    [model_input["file_path_video"]] if model_input["file_path_video"] else []
                ),
                "thermal_paths": [],
                "do_sample": False,
                "top_p": 0.9,
                "temperature": 1.0,
                "max_tgt_len": 30,
                "modality_embeds": [],
            }
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
        elif prompt_type in [
            "audio",
            "overlayed_full_audio_classification",
            "simple_audio_classification",
        ]:
            video_file_path, audio_file_path = None, file_path
        elif prompt_type in ["temporal_video", "silent_video", "video_segment"]:
            video_file_path, audio_file_path = file_path, None
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
        }

        return model_input
