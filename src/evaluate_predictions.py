#!/usr/bin/env python3

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from utils.constants import QUESTION_TYPES, PROMPT_TYPES, TASKS


MAPPER = {
    'joint': 'DAVE',
    'regular': 'Multimodal synchronisation',
    'none_of_the_above_no_sound': 'Sound absence detection',
    'none_of_the_above_incorrect_audio': 'Sound discrimination',
    'multimodal': 'Multimodal synchronisation',
    'temporal_video': 'Temporal ordering',
    'overlayed_full_audio_classification': 'Audio classification',
    'video_segment': 'Action recognition',
    'silent_video': 'Video + Text',
    'audio': 'Audio + Text',
    'text_only': 'Text'
}


def format_mean_std(value, std=None):
    if std is None:
        return f"${(100 * value):.2f}$"
    return f"${(100 * value):.2f}_{{\\pm {(100 * std):.2f}}}$"


def bootstrap_accuracy(is_correct_list, n_bootstrap=1000):
    """Perform bootstrap resampling to estimate accuracy distribution."""
    n = len(is_correct_list)
    bootstrap_acc = []
    for _ in range(n_bootstrap):
        idxs = np.random.randint(0, n, size=n)  # Sample with replacement
        acc = np.mean(is_correct_list[idxs])
        bootstrap_acc.append(acc)
    return np.array(bootstrap_acc)


def print_question_types_results(data, question_types):
    model_names = data.keys()
    for model_name in model_names:
        print(f"\nResults for model: {model_name}")
        model_data = data[model_name]['multimodal']
        for type in question_types:
            if type == 'joint':
                is_correct_list = np.array([sample["is_correct"] for sample in model_data])
            else:
                is_correct_list = np.array([
                    sample["is_correct"] for sample in model_data if sample['question_type'] == type
                ])
            if len(is_correct_list) == 0:
                print(f"  {MAPPER[type]}: No samples")
                continue
            bootstrap_acc = bootstrap_accuracy(is_correct_list)
            print(f"  {MAPPER[type]}: {format_mean_std(np.mean(bootstrap_acc), np.std(bootstrap_acc))}")

def print_results(data, subset):
    for model_name, model_data in data.items():
        print(f"\nResults for model: {model_name}")
        for element in subset:
            if element in model_data:
                is_correct_list = np.array([sample["is_correct"] for sample in model_data[element]])
            else:
                continue

            bootstrap_acc = bootstrap_accuracy(is_correct_list)
            print(f" {MAPPER[element]}: {format_mean_std(np.mean(bootstrap_acc), np.std(bootstrap_acc))} ")


def merge_json_files(directories):
    merged_predictions = defaultdict(lambda: defaultdict(list))

    for directory in directories:
        if os.path.isdir(directory):
            for filename in os.listdir(directory):
                if filename.endswith('.json'):
                    filepath = os.path.join(directory, filename)
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        predictions = data.get("predictions", {})
                        # Iterate through the predictions and merge them by model name and prompt type
                        for model_name, model_data in predictions.items():
                            for prompt_type, prediction_list in model_data.items():
                                merged_predictions[model_name][prompt_type].extend(prediction_list)
        else:
            print(f"Warning: {directory} is not a valid directory.")

    return merged_predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction accuracy from JSON file.")
    parser.add_argument("--predictions_file", type=str, help="Path to predictions JSON file")
    parser.add_argument("--result_dir", type=str, nargs='+', help="Path to result dir.")
    args = parser.parse_args()

    if args.predictions_file:
        with open(args.predictions_file, "r") as f:
            all_predictions = json.load(f)

        print_question_types_results(all_predictions['predictions'], QUESTION_TYPES)

    elif args.result_dir:
        merged_predictions = merge_json_files(args.result_dir)

        # Merge the Ego4D and Epic inference files
        merged_data = {"predictions": merged_predictions}

        data = merged_data['predictions']
        print_question_types_results(data, QUESTION_TYPES)
        print(70 * '*')
        print_results(data, TASKS)
        print(70 * '*')
        print_results(data, PROMPT_TYPES)
    else:
        raise ValueError("Either --predictions_file or --result_dir must be provided.")


if __name__ == "__main__":
    main()