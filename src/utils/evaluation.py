class Evaluator:

    def __init__(self, prompt_type: str):
        self.prompt_type = prompt_type
        self.stats = {"correct": 0, "total": 0}

    def is_correct(self, prediction, ground_truth):
        if self.prompt_type == "temporal_video":
            if len(prediction) != len(ground_truth):
                return False
            return all(prediction[idx] == ground_truth[idx] for idx in range(len(ground_truth)))
        return prediction in ground_truth

    def process_sample(self, prediction, ground_truth):
        is_correct = self.is_correct(prediction, ground_truth)
        self.stats["correct"] += is_correct
        self.stats["total"] += 1
        return is_correct

    def evaluate(self):
        return self.stats["correct"] / self.stats["total"] * 100 if self.stats["total"] else 0

    @staticmethod
    def compute_timestamp_accuracy(
        pred_timestamp_start,
        pred_timestamp_end,
        gt_timestamp_start,
        gt_timestamp_end,
        delta=1.0,
    ):
        # Compute Intersection over Union (IoU)
        intersection = max(
            0,
            min(pred_timestamp_end, gt_timestamp_end)
            - max(pred_timestamp_start, gt_timestamp_start),
        )
        union = (
            (pred_timestamp_end - pred_timestamp_start)
            + (gt_timestamp_end - gt_timestamp_start)
            - intersection
        )
        iou = intersection / union if union > 0 else 0

        # Check if prediction is within delta tolerance
        tolerance_accuracy = int(
            abs(pred_timestamp_start - gt_timestamp_start) <= delta
            and abs(pred_timestamp_end - gt_timestamp_end) <= delta
        )

        return {"iou": iou, "tolerance_accuracy": tolerance_accuracy}
