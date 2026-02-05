"""
VQA Evaluation for AnyMAL

Evaluates the model on Visual Question Answering benchmarks.

Supported benchmarks:
- VQAv2: Visual Question Answering v2.0
- TextVQA: VQA with text in images (OCR)
- OKVQA: Outside Knowledge VQA
- ScienceQA: Science domain questions

Educational Notes:
-----------------
VQA Evaluation Protocol:
1. Load question + image pairs
2. Generate model answer
3. Compare with ground truth
4. Compute accuracy

VQAv2 uses soft accuracy:
- Multiple human annotations per question
- Answer is correct if at least 3/10 annotators agree
- Accuracy = min(1, count(answer) / 3)

This handles answer variation:
- "dog" vs "a dog" vs "dogs" all may be correct
- Human agreement provides soft labels
"""

import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any, List
from tqdm import tqdm
from collections import Counter


class VQADataset(Dataset):
    """
    Dataset for VQA evaluation.

    Args:
        questions_file: Path to questions JSON
        annotations_file: Path to annotations JSON (for ground truth)
        image_dir: Directory containing images
        transform: Image transform
        tokenizer: LLaMA tokenizer
    """

    def __init__(
        self,
        questions_file: str,
        annotations_file: Optional[str],
        image_dir: str,
        transform,
        tokenizer,
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer

        # Load questions
        with open(questions_file, "r") as f:
            questions_data = json.load(f)
        self.questions = questions_data["questions"]

        # Load annotations if available
        self.annotations = {}
        if annotations_file and os.path.exists(annotations_file):
            with open(annotations_file, "r") as f:
                annotations_data = json.load(f)
            for ann in annotations_data["annotations"]:
                self.annotations[ann["question_id"]] = ann

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        q = self.questions[idx]

        # Load image
        image_id = q["image_id"]
        image_path = os.path.join(
            self.image_dir,
            f"COCO_val2014_{image_id:012d}.jpg"  # VQAv2 uses COCO val2014
        )

        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        # Get question
        question = q["question"]
        question_id = q["question_id"]

        # Format prompt
        prompt = f"Question: {question}\nAnswer:"

        # Tokenize
        encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=256,
        )

        # Get ground truth if available
        answers = []
        if question_id in self.annotations:
            ann = self.annotations[question_id]
            answers = [a["answer"] for a in ann["answers"]]

        return {
            "image": image_tensor,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "question_id": question_id,
            "question": question,
            "answers": answers,
        }


class VQAEvaluator:
    """
    Evaluator for VQA benchmarks.

    Args:
        model: AnyMAL model
        device: Device to run on
        max_new_tokens: Maximum tokens to generate
    """

    def __init__(
        self,
        model,
        device: torch.device = None,
        max_new_tokens: int = 32,
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        output_file: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate on VQA dataset.

        Args:
            dataloader: DataLoader for VQA dataset
            output_file: Optional file to save predictions

        Returns:
            Dict with accuracy metrics
        """
        predictions = []
        total_score = 0.0
        num_samples = 0

        for batch in tqdm(dataloader, desc="Evaluating VQA"):
            # Move to device
            images = batch["image"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # Generate answers
            generated_ids = self.model.generate(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Greedy decoding for evaluation
            )

            # Decode predictions
            for i in range(len(generated_ids)):
                # Get only the generated part
                prompt_len = input_ids[i].shape[0]
                # When generating from `inputs_embeds` (multimodal), HF may return only new tokens.
                seq = generated_ids[i]
                generated = seq[prompt_len:] if seq.shape[0] > prompt_len else seq

                # Decode
                pred_answer = self.model.tokenizer.decode(
                    generated,
                    skip_special_tokens=True,
                ).strip()

                # Clean up answer
                pred_answer = self._process_answer(pred_answer)

                question_id = batch["question_id"][i].item() if isinstance(
                    batch["question_id"][i], torch.Tensor
                ) else batch["question_id"][i]

                predictions.append({
                    "question_id": question_id,
                    "answer": pred_answer,
                })

                # Compute score if ground truth available
                gt_answers = batch["answers"][i] if batch["answers"] else []
                if gt_answers:
                    score = self._compute_vqa_score(pred_answer, gt_answers)
                    total_score += score
                    num_samples += 1

        # Compute overall accuracy
        accuracy = total_score / num_samples if num_samples > 0 else 0.0

        # Save predictions
        if output_file:
            with open(output_file, "w") as f:
                json.dump(predictions, f, indent=2)

        return {
            "accuracy": accuracy * 100,  # Percentage
            "num_samples": num_samples,
        }

    def _process_answer(self, answer: str) -> str:
        """
        Process generated answer for evaluation.

        VQA evaluation uses processed answers:
        - Lowercase
        - Remove punctuation
        - Remove articles (a, an, the)
        """
        # Take first line/sentence
        answer = answer.split("\n")[0].split(".")[0]

        # Lowercase
        answer = answer.lower().strip()

        # Remove common prefixes
        prefixes = ["the answer is", "answer:", "it is", "this is"]
        for prefix in prefixes:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()

        # Remove punctuation
        import re
        answer = re.sub(r"[^\w\s]", "", answer)

        # Remove articles
        articles = ["a", "an", "the"]
        words = answer.split()
        words = [w for w in words if w not in articles]
        answer = " ".join(words)

        return answer.strip()

    def _compute_vqa_score(
        self,
        prediction: str,
        ground_truths: List[str],
    ) -> float:
        """
        Compute VQA accuracy score.

        VQA uses soft accuracy:
        accuracy = min(1, count(answer) / 3)

        If an answer appears in at least 3 of 10 annotations,
        it's considered fully correct.
        """
        # Process ground truth answers
        processed_gts = [self._process_answer(gt) for gt in ground_truths]

        # Count matching answers
        count = sum(1 for gt in processed_gts if prediction == gt)

        # VQA accuracy formula
        return min(1.0, count / 3)


def vqa_collate_fn(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    """
    Custom collate function for VQA evaluation with variable-length sequences.

    Args:
        batch: List of sample dicts from VQADataset
        pad_token_id: Token ID for padding

    Returns:
        Batched dict with padded tensors
    """
    images = torch.stack([item["image"] for item in batch])
    max_len = max(item["input_ids"].shape[-1] for item in batch)

    input_ids_list = []
    attention_masks_list = []

    for item in batch:
        ids = item["input_ids"]
        mask = item["attention_mask"]
        pad_len = max_len - ids.shape[0]
        input_ids_list.append(F.pad(ids, (0, pad_len), value=pad_token_id))
        attention_masks_list.append(F.pad(mask, (0, pad_len), value=0))

    return {
        "image": images,
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_masks_list),
        "question_id": [item["question_id"] for item in batch],
        "question": [item["question"] for item in batch],
        "answers": [item["answers"] for item in batch],
    }


def evaluate_vqav2(
    model,
    questions_file: str,
    annotations_file: str,
    image_dir: str,
    output_file: Optional[str] = None,
    batch_size: int = 16,
) -> Dict[str, float]:
    """
    Convenience function to evaluate on VQAv2.

    Args:
        model: AnyMAL model
        questions_file: Path to VQAv2 questions
        annotations_file: Path to VQAv2 annotations
        image_dir: Path to COCO images
        output_file: Optional output file for predictions
        batch_size: Evaluation batch size

    Returns:
        Evaluation metrics
    """
    from data import get_image_transform

    transform = get_image_transform(image_size=224, is_train=False)

    dataset = VQADataset(
        questions_file=questions_file,
        annotations_file=annotations_file,
        image_dir=image_dir,
        transform=transform,
        tokenizer=model.tokenizer,
    )

    pad_token_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda b: vqa_collate_fn(b, pad_token_id),
    )

    evaluator = VQAEvaluator(model)
    results = evaluator.evaluate(dataloader, output_file)

    print(f"\nVQAv2 Results:")
    print(f"  Accuracy: {results['accuracy']:.2f}%")
    print(f"  Samples: {results['num_samples']}")

    return results
