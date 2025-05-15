#!/usr/bin/env python3

import json
import argparse
import numpy as np
from utils.constants import CATEGORIES


CATEGORIES_MAPPER = {
    'joint': 'DAVE',
    'regular': 'Multimodal synchronisation',
    'none_of_the_above_no_sound': 'Sound absence detection',
    'none_of_the_above_incorrect_audio': 'Sound discrimination'
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


def print_categories_results(data, categories):
    model_names = data.keys()
    for model_name in model_names:
        print(f"\nResults for model: {model_name}")
        model_data = data[model_name]['multimodal']
        for category in categories:
            if category == 'joint':
                is_correct_list = np.array([sample["is_correct"] for sample in model_data])
            else:
                is_correct_list = np.array([
                    sample["is_correct"] for sample in model_data if sample['question_type'] == category
                ])
            if len(is_correct_list) == 0:
                print(f"  {CATEGORIES_MAPPER[category]}: No samples")
                continue
            bootstrap_acc = bootstrap_accuracy(is_correct_list)
            print(f"  {CATEGORIES_MAPPER[category]}: {format_mean_std(np.mean(bootstrap_acc), np.std(bootstrap_acc))}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction accuracy from JSON file.")
    parser.add_argument("--predictions_file", type=str, help="Path to predictions JSON file")
    args = parser.parse_args()

    with open(args.predictions_file, "r") as f:
        all_predictions = json.load(f)

    print_categories_results(all_predictions['predictions'], CATEGORIES)


if __name__ == "__main__":
    main()