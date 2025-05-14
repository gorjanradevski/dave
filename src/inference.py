import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from datasets.arrow_dataset import Dataset as HuggingfaceDataset
from tqdm.auto import tqdm

from models import model_factory
from utils.constants import model_choices
from utils.data import DatasetWrapper
from utils.evaluation import Evaluator


def filter_available_samples(
    huggingface_dataset: HuggingfaceDataset, google_id_mapping: Dict[str, str]
) -> List[Dict]:
    """Filter samples to only include those with available files in google_id_mapping."""
    filtered_samples = []

    for sample in tqdm(huggingface_dataset, desc="Validating dataset samples"):
        # Check all possible file paths for this sample
        path_mapping = {
            "multimodal": "video_with_overlayed_audio_path",
            "audio": "overlayed_audio_path",
            "silent_video": "silent_video_path",
            "overlayed_full_audio_classification": "overlayed_audio_path",
            "simple_audio_classification": "audio_path",
            "temporal_video": "video_with_overlayed_audio_path",
            "video_segment": "event_video_path",
        }

        # Check if all required files are available
        files_available = True
        for path_key in path_mapping.values():
            if path_key not in sample:
                continue
            file_path = str(Path(sample[path_key]))
            if file_path not in google_id_mapping:
                files_available = False
                break

        if files_available:
            filtered_samples.append(sample)

    print(f"Filtered dataset: {len(filtered_samples)}/{len(huggingface_dataset)} samples available")
    return filtered_samples


def filter_by_prompt(data_samples: List[Dict[str, Any]], prompt_type: str):
    filtered_samples = []
    for sample in data_samples:
        # If the same is none of the above, and the prompt is not one of the core tasks, we filter
        if sample["type"] != "regular" and prompt_type in [
            "video_segment",
            "temporal_video",
            "pipeline_event_classification",
            "overlayed_full_audio_classification",
            "simple_audio_classification",
        ]:
            continue
        filtered_samples.append(sample)
    return filtered_samples


def parse_arguments() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(description="Process and analyze video-based datasets.")
    parser.add_argument("--split", type=str, required=True, help="Split name: epic or ego4d.")
    parser.add_argument("--google_id_mapping_path", type=str)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        default=["gemini-1.5-flash-latest"],
        help=f"Name of the Generative AI model: {model_choices}",
    )
    parser.add_argument(
        "--prompt_types",
        nargs="+",
        default=["multimodal"],
        help="The type of prompt sto use.",
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    # Check model names
    if "pipeline_event_classification" in args.prompt_types:
        if "gemini-pipeline" not in args.model_names and "openai" not in args.model_names:
            raise ValueError(
                "Pipeline event classification requires a pipeline model: gemini-pipeline or openai"
            )

    # Load and prepare dataset
    huggingface_dataset = load_dataset(
        "gorjanradevski/dave",
        split=args.split,
        keep_in_memory=True,
        trust_remote_code=True,
    )

    google_id_mapping = None
    if args.google_id_mapping_path is not None:
        google_id_mapping = json.load(open(args.google_id_mapping_path, "r"))

    if google_id_mapping is not None:
        data_samples = filter_available_samples(huggingface_dataset, google_id_mapping)

    if not data_samples:
        raise ValueError("No valid samples found after filtering!")

    print("*" * 50)
    print(f"Trying out models: {args.model_names}")
    print(f"Using prompt types: {args.prompt_types}")
    print(f"Processing dataset of size: {len(data_samples)}")
    print(f"Dataset split for DAVE: {args.split}")
    print("*" * 50)

    latex_prompt_types = " & ".join(args.prompt_types)

    all_predictions = {
        "predictions": {mn: {p: [] for p in args.prompt_types} for mn in args.model_names},
    }

    for model_name in args.model_names:
        print(f"Processing model: {model_name}")
        model = model_factory(model_name=model_name)(
            model_name=model_name, google_id_mapping=google_id_mapping
        )

        latex_row = f"{model_name} "
        for prompt_type in args.prompt_types:

            data_samples = filter_by_prompt(data_samples, prompt_type=prompt_type)

            dataset = DatasetWrapper(
                data_samples=data_samples,
                prompt_type=prompt_type,
            )
            evaluator = Evaluator(prompt_type=prompt_type)

            print(f"Processing prompt type: {prompt_type}")
            for sample in tqdm(dataset):
                try:
                    model_input = model.prepare_model_input(
                        sample=sample,
                        prompt_suffix=dataset.prompt_suffix,
                    )
                    response_dict = model.inference(
                        model_input=model_input, prompt_type=prompt_type
                    )
                    is_correct = evaluator.process_sample(
                        prediction=response_dict["response_text"],
                        ground_truth=sample["ground_truth"],
                    )
                    timestamps_acc_dict = {"iou": -1, "tolerance_accuracy": -1}
                    if sample["type"] == "regular" and "predicted_timestamps" in response_dict:
                        timestamps_acc_dict = evaluator.compute_timestamp_accuracy(
                            *response_dict["predicted_timestamps"],
                            *sample["ground_truth_timestamps"],
                        )

                    all_predictions["predictions"][model_name][prompt_type].append(
                        {
                            "response_dict": response_dict,
                            "ground_truth": sample["ground_truth"],
                            "prompt": model_input["prompt"],
                            "question_type": sample.get("type"),
                            "is_correct": is_correct,
                            "timestamp_iou": timestamps_acc_dict["iou"],
                            "timestamp_acc": timestamps_acc_dict["tolerance_accuracy"],
                            "overlayed_event_index": sample.get("overlayed_event_index", None),
                        }
                    )
                except Exception as e:
                    print(f"Skipping sample due to error: {e}")
                    continue  # Skip this sample and move to the next one

            # Calculate and display results
            accuracy = evaluator.evaluate()
            print(f"Accuracy: {accuracy:.2f}%")
            latex_row += f"& {accuracy:.2f} "

        latex_row += "\\\\"

        print("*" * 50)
        print(latex_prompt_types)
        print(latex_row)
        print("*" * 50)


if __name__ == "__main__":
    main()
