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


TRAINING_SYSTEM_PROMPT = (
    "You are a helpful AI assistant that can see and understand images. "
    "Provide detailed, accurate, and helpful responses to questions about images."
)
SYSTEM_HEADER = "<|start_header_id|>system<|end_header_id|>\n\n"
USER_HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"
END_TURN = "<|eot_id|>"
IMAGE_SENTINEL = "<|image_sentinel|>"


def _special_stop_token_ids(tokenizer) -> List[int]:
    stop_ids = []
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        stop_ids.append(int(eos_id))
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        eot_id = get_vocab().get(END_TURN)
        if eot_id is not None:
            stop_ids.append(int(eot_id))
    return list(dict.fromkeys(stop_ids))


def _trim_at_stop_token(token_ids: torch.Tensor, stop_token_ids: List[int]) -> torch.Tensor:
    if token_ids.numel() == 0 or not stop_token_ids:
        return token_ids
    stop = set(stop_token_ids)
    for idx, token_id in enumerate(token_ids.tolist()):
        if int(token_id) in stop:
            return token_ids[: idx + 1]
    return token_ids


def _build_chat_prompt_ids(
    tokenizer,
    question: str,
    image_placeholder_token_id: Optional[int],
    num_image_tokens: int,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    user_text = question.strip()
    if image_placeholder_token_id is not None and num_image_tokens > 0:
        user_text = f"{IMAGE_SENTINEL}\n{user_text}"
    text = (
        f"{SYSTEM_HEADER}{TRAINING_SYSTEM_PROMPT}{END_TURN}"
        f"{USER_HEADER}{user_text}{END_TURN}"
        f"{ASSISTANT_HEADER}"
    )
    encoding = tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    input_ids = encoding["input_ids"].squeeze(0)
    attention_mask = encoding["attention_mask"].squeeze(0)

    if image_placeholder_token_id is None or num_image_tokens <= 0:
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    offsets = encoding["offset_mapping"].squeeze(0)
    sentinel_start = text.find(IMAGE_SENTINEL)
    sentinel_end = sentinel_start + len(IMAGE_SENTINEL)
    sentinel_indices = [
        i
        for i, (token_start, token_end) in enumerate(offsets.tolist())
        if token_end > sentinel_start and token_start < sentinel_end
    ]
    if not sentinel_indices:
        raise RuntimeError("Could not locate image sentinel after tokenization")

    first = sentinel_indices[0]
    last = sentinel_indices[-1] + 1
    placeholders = torch.full(
        (int(num_image_tokens),),
        int(image_placeholder_token_id),
        dtype=input_ids.dtype,
    )
    placeholder_mask = torch.ones(int(num_image_tokens), dtype=attention_mask.dtype)
    input_ids = torch.cat([input_ids[:first], placeholders, input_ids[last:]])
    attention_mask = torch.cat([attention_mask[:first], placeholder_mask, attention_mask[last:]])
    if input_ids.shape[0] > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
    return {"input_ids": input_ids, "attention_mask": attention_mask}


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
        filter_to_available_images: bool = True,
        image_placeholder_token_id: Optional[int] = None,
        num_image_tokens: int = 0,
        prompt_style: str = "training_chat",
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.image_placeholder_token_id = image_placeholder_token_id
        self.num_image_tokens = int(num_image_tokens or 0)
        self.prompt_style = prompt_style

        # Load questions
        with open(questions_file, "r") as f:
            questions_data = json.load(f)
        self.questions = questions_data["questions"]
        self.original_num_questions = len(self.questions)

        # Load annotations if available
        self.annotations = {}
        self.annotation_meta = {}
        if annotations_file and os.path.exists(annotations_file):
            with open(annotations_file, "r") as f:
                annotations_data = json.load(f)
            for ann in annotations_data["annotations"]:
                self.annotations[ann["question_id"]] = ann
                self.annotation_meta[ann["question_id"]] = {
                    "answer_type": ann.get("answer_type", ""),
                    "question_type": ann.get("question_type", ""),
                }

        # Filter to only questions whose images exist on disk
        self.missing_image_count = 0
        if filter_to_available_images and image_dir and os.path.exists(image_dir):
            available_files = set(f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png')))
            total = len(self.questions)
            self.questions = [
                q for q in self.questions
                if f"COCO_val2014_{q['image_id']:012d}.jpg" in available_files
            ]
            self.missing_image_count = total - len(self.questions)
            print(f"VQA filtered to {len(self.questions)}/{total} questions with available images")

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        q = self.questions[idx]

        # Load image
        image_id = q["image_id"]
        image_path = os.path.join(
            self.image_dir,
            f"COCO_val2014_{image_id:012d}.jpg"  # VQAv2 uses COCO val2014
        )

        from PIL import Image
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image)
        except (FileNotFoundError, OSError) as e:
            print(f"VQA: skipping image {image_path}: {e}")
            return None

        # Get question
        question = q["question"]
        question_id = q["question_id"]

        if self.prompt_style == "training_chat":
            encoded = _build_chat_prompt_ids(
                tokenizer=self.tokenizer,
                question=question,
                image_placeholder_token_id=self.image_placeholder_token_id,
                num_image_tokens=self.num_image_tokens,
                max_length=768,
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
        elif self.prompt_style == "legacy_qa":
            prompt = f"Question: {question}\nAnswer:"
            encoding = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=256,
            )
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)

            if self.image_placeholder_token_id is not None and self.num_image_tokens > 0:
                placeholders = torch.full(
                    (self.num_image_tokens,),
                    self.image_placeholder_token_id,
                    dtype=input_ids.dtype,
                )
                placeholder_mask = torch.ones(self.num_image_tokens, dtype=attention_mask.dtype)
                input_ids = torch.cat([placeholders, input_ids], dim=0)
                attention_mask = torch.cat([placeholder_mask, attention_mask], dim=0)
        else:
            raise ValueError(f"Unknown VQA prompt_style: {self.prompt_style}")

        # Get ground truth if available
        answers = []
        answer_type = ""
        question_type = ""
        if question_id in self.annotations:
            ann = self.annotations[question_id]
            answers = [a["answer"] for a in ann["answers"]]
            meta = self.annotation_meta.get(question_id, {})
            answer_type = meta.get("answer_type", "")
            question_type = meta.get("question_type", "")

        return {
            "image": image_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "question_id": question_id,
            "question": question,
            "answers": answers,
            "answer_type": answer_type,
            "question_type": question_type,
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
        self.stop_token_ids = _special_stop_token_ids(self.model.tokenizer)

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
        generated_token_counts = []
        clean_eos_count = 0
        hit_max_new_tokens_count = 0
        answer_type_scores = {}

        for batch in tqdm(dataloader, desc="Evaluating VQA"):
            if batch is None:
                continue

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
                generated = _trim_at_stop_token(generated, self.stop_token_ids)
                generated_token_counts.append(int(generated.shape[0]))
                hit_stop = generated.numel() > 0 and int(generated[-1].item()) in set(self.stop_token_ids)
                if hit_stop:
                    clean_eos_count += 1
                elif generated.shape[0] >= self.max_new_tokens:
                    hit_max_new_tokens_count += 1

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
                    answer_type = ""
                    if "answer_type" in batch:
                        answer_type = batch["answer_type"][i] or ""
                    if answer_type:
                        bucket = answer_type_scores.setdefault(answer_type, {"score": 0.0, "count": 0})
                        bucket["score"] += score
                        bucket["count"] += 1

        # Compute overall accuracy
        accuracy = total_score / num_samples if num_samples > 0 else 0.0

        # Save predictions
        if output_file:
            with open(output_file, "w") as f:
                json.dump(predictions, f, indent=2)

        avg_generated_tokens = (
            sum(generated_token_counts) / len(generated_token_counts)
            if generated_token_counts else 0.0
        )
        eos_rate = clean_eos_count / len(generated_token_counts) if generated_token_counts else 0.0
        hit_max_new_tokens_rate = (
            hit_max_new_tokens_count / len(generated_token_counts) if generated_token_counts else 0.0
        )

        results = {
            "accuracy": accuracy * 100,  # Percentage
            "num_samples": num_samples,
            "avg_generated_tokens": avg_generated_tokens,
            "eos_rate": eos_rate,
            "hit_max_new_tokens_rate": hit_max_new_tokens_rate,
        }
        for answer_type, bucket in sorted(answer_type_scores.items()):
            key = answer_type.replace("/", "_").replace(" ", "_")
            results[f"accuracy_{key}"] = (
                100.0 * bucket["score"] / bucket["count"] if bucket["count"] else 0.0
            )
            results[f"num_samples_{key}"] = bucket["count"]

        return results

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


def vqa_collate_fn(batch: List[Dict[str, Any]], pad_token_id: int) -> Optional[Dict[str, Any]]:
    """
    Custom collate function for VQA evaluation with variable-length sequences.

    Args:
        batch: List of sample dicts from VQADataset
        pad_token_id: Token ID for padding

    Returns:
        Batched dict with padded tensors, or None if all samples are invalid
    """
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
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
        "answer_type": [item.get("answer_type", "") for item in batch],
        "question_type": [item.get("question_type", "") for item in batch],
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
    from data import get_vision_transform

    is_siglip_arch = getattr(model, "architecture", "") in {"anymal_v2", "anymal_v3"}
    vision_type = getattr(model, "vision_encoder_type", "clip")
    vision_model = getattr(getattr(model, "image_encoder", None), "model_name", None)
    transform = get_vision_transform(
        vision_encoder_type=vision_type,
        vision_model_name=vision_model,
        image_size=384 if is_siglip_arch else 224,
        is_train=False,
        use_augmentation=False,
    )

    dataset = VQADataset(
        questions_file=questions_file,
        annotations_file=annotations_file,
        image_dir=image_dir,
        transform=transform,
        tokenizer=model.tokenizer,
        image_placeholder_token_id=getattr(model, "image_placeholder_token_id", None),
        num_image_tokens=getattr(model, "num_image_tokens", 0) if is_siglip_arch else 0,
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
