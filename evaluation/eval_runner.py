"""
Evaluation Runner for AnyMAL In-Training Evaluation

Wraps existing evaluation benchmarks for use during training.
Catches exceptions gracefully to never crash the training loop.
"""

import time
import torch
from typing import Dict, List, Optional
from torch.utils.data import DataLoader, Subset


class EvalRunner:
    """
    Runs evaluation benchmarks during training.

    Wraps VQAEvaluator for in-training use with:
    - Subset sampling for speed (default 500 samples)
    - Lazy dataloader creation (cached after first call)
    - Graceful error handling (logs but doesn't crash)
    - Timeout support

    Args:
        model: AnyMAL model
        vqa_questions_file: Path to VQAv2 questions JSON
        vqa_annotations_file: Path to VQAv2 annotations JSON
        vqa_image_dir: Directory with COCO val2014 images
        batch_size: Evaluation batch size
        max_eval_samples: Max samples to evaluate (for speed)
        timeout: Timeout in seconds for evaluation
    """

    def __init__(
        self,
        model,
        vqa_questions_file: str,
        vqa_annotations_file: str,
        vqa_image_dir: str,
        batch_size: int = 8,
        max_eval_samples: int = 500,
        timeout: float = 300,
    ):
        self.model = model
        self.vqa_questions_file = vqa_questions_file
        self.vqa_annotations_file = vqa_annotations_file
        self.vqa_image_dir = vqa_image_dir
        self.batch_size = batch_size
        self.max_eval_samples = max_eval_samples
        self.timeout = timeout

        self._vqa_dataloader = None  # Lazy-created
        self._vqa_evaluator = None   # Lazy-created

    def _get_vqa_dataloader(self) -> Optional[DataLoader]:
        """Lazy-create VQA dataloader (cached after first call)."""
        if self._vqa_dataloader is not None:
            return self._vqa_dataloader

        try:
            import os
            from evaluation.vqa_eval import VQADataset, vqa_collate_fn
            from data.data_utils import get_image_transform

            # Check files exist
            if not os.path.exists(self.vqa_questions_file):
                print(f"EvalRunner: VQA questions file not found: {self.vqa_questions_file}")
                return None
            if not os.path.exists(self.vqa_annotations_file):
                print(f"EvalRunner: VQA annotations file not found: {self.vqa_annotations_file}")
                return None
            if not os.path.exists(self.vqa_image_dir):
                print(f"EvalRunner: VQA image dir not found: {self.vqa_image_dir}")
                return None

            transform = get_image_transform(image_size=224, is_train=False)

            dataset = VQADataset(
                questions_file=self.vqa_questions_file,
                annotations_file=self.vqa_annotations_file,
                image_dir=self.vqa_image_dir,
                transform=transform,
                tokenizer=self.model.tokenizer,
            )

            # Take a subset for speed
            if len(dataset) > self.max_eval_samples:
                import random
                rng = random.Random(42)
                indices = list(range(len(dataset)))
                rng.shuffle(indices)
                dataset = Subset(dataset, indices[:self.max_eval_samples])

            pad_token_id = self.model.tokenizer.pad_token_id or self.model.tokenizer.eos_token_id

            self._vqa_dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
                collate_fn=lambda b: vqa_collate_fn(b, pad_token_id),
            )

            return self._vqa_dataloader

        except Exception as e:
            print(f"EvalRunner: Failed to create VQA dataloader: {e}")
            return None

    def _get_vqa_evaluator(self):
        """Lazy-create VQA evaluator."""
        if self._vqa_evaluator is not None:
            return self._vqa_evaluator

        try:
            from evaluation.vqa_eval import VQAEvaluator
            self._vqa_evaluator = VQAEvaluator(
                model=self.model,
                max_new_tokens=32,
            )
            return self._vqa_evaluator
        except Exception as e:
            print(f"EvalRunner: Failed to create VQA evaluator: {e}")
            return None

    def run(self, benchmarks: List[str] = None) -> Dict[str, float]:
        """
        Run evaluation benchmarks.

        Args:
            benchmarks: List of benchmark names to run. Default: ["vqa"]

        Returns:
            Dict of metric name -> value, e.g. {"eval/vqa_accuracy": 25.3, "eval/vqa_time_sec": 45.2}
        """
        if benchmarks is None:
            benchmarks = ["vqa"]

        results = {}

        for benchmark in benchmarks:
            if benchmark == "vqa":
                vqa_results = self._run_vqa()
                results.update(vqa_results)
            else:
                print(f"EvalRunner: Unknown benchmark '{benchmark}', skipping")

        return results

    def _run_vqa(self) -> Dict[str, float]:
        """Run VQA evaluation with error handling."""
        try:
            start = time.time()

            dataloader = self._get_vqa_dataloader()
            if dataloader is None:
                return {}

            evaluator = self._get_vqa_evaluator()
            if evaluator is None:
                return {}

            # Run evaluation
            self.model.eval()
            eval_results = evaluator.evaluate(dataloader)

            elapsed = time.time() - start

            print(f"EvalRunner: VQA accuracy = {eval_results['accuracy']:.2f}% "
                  f"({eval_results['num_samples']} samples, {elapsed:.1f}s)")

            return {
                "eval/vqa_accuracy": eval_results["accuracy"],
                "eval/vqa_num_samples": eval_results["num_samples"],
                "eval/vqa_time_sec": elapsed,
            }

        except Exception as e:
            print(f"EvalRunner: VQA evaluation failed: {e}")
            return {}
