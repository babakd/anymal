"""
Captioning Evaluation for AnyMAL

Evaluates the model on image captioning benchmarks using COCO-style metrics.

Metrics:
- BLEU (1-4): Precision-based n-gram matching
- METEOR: Harmonic mean of precision and recall with synonyms
- ROUGE-L: Longest common subsequence
- CIDEr: Consensus-based metric using TF-IDF
- SPICE: Semantic similarity using scene graphs

Educational Notes:
-----------------
CIDEr (Consensus-based Image Description Evaluation):
- Designed specifically for image captioning
- Uses TF-IDF weighting to prioritize informative words
- Computes cosine similarity between n-gram vectors
- Higher weight for rare but accurate descriptions

Why CIDEr is preferred:
- BLEU favors short outputs (precision-based)
- CIDEr rewards both precision AND information content
- Better correlation with human judgment for captions

Typical good scores on COCO:
- CIDEr: 100-130 (human ~85, SOTA ~145)
- BLEU-4: 35-40
- METEOR: 25-30
"""

import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any, List
from tqdm import tqdm


class COCOCaptionDataset(Dataset):
    """
    Dataset for COCO captioning evaluation.

    Args:
        annotations_file: Path to COCO captions annotations
        image_dir: Directory containing images
        transform: Image transform
        tokenizer: LLaMA tokenizer
        split: Dataset split (train, val, test)
    """

    def __init__(
        self,
        annotations_file: str,
        image_dir: str,
        transform,
        tokenizer,
        split: str = "val",
        filter_to_available_images: bool = True,
        image_placeholder_token_id: Optional[int] = None,
        num_image_tokens: int = 0,
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.split = split
        self.image_placeholder_token_id = image_placeholder_token_id
        self.num_image_tokens = int(num_image_tokens or 0)

        # Load annotations
        with open(annotations_file, "r") as f:
            coco_data = json.load(f)

        # Build image_id -> captions mapping
        self.images = coco_data["images"]
        self.annotations = {}
        for ann in coco_data["annotations"]:
            image_id = ann["image_id"]
            if image_id not in self.annotations:
                self.annotations[image_id] = []
            self.annotations[image_id].append(ann["caption"])

        if filter_to_available_images and image_dir and os.path.exists(image_dir):
            total = len(self.images)
            self.images = [
                img for img in self.images
                if os.path.exists(os.path.join(image_dir, img["file_name"]))
            ]
            print(
                f"COCO captioning filtered to {len(self.images)}/{total} "
                "images with available files"
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        img_info = self.images[idx]
        image_id = img_info["id"]
        file_name = img_info["file_name"]

        # Load image
        image_path = os.path.join(self.image_dir, file_name)

        from PIL import Image
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image)
        except (FileNotFoundError, OSError) as e:
            print(f"COCO captioning: skipping image {image_path}: {e}")
            return None

        # Create prompt for captioning
        prompt = "Describe this image in detail:"

        # Tokenize
        encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=64,
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

        # Get ground truth captions
        captions = self.annotations.get(image_id, [])

        return {
            "image": image_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_id": image_id,
            "captions": captions,
        }


class CaptioningEvaluator:
    """
    Evaluator for image captioning.

    Uses pycocoevalcap for computing metrics.

    Args:
        model: AnyMAL model
        device: Device to run on
        max_new_tokens: Maximum tokens to generate
    """

    def __init__(
        self,
        model,
        device: torch.device = None,
        max_new_tokens: int = 64,
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
        Evaluate captioning performance.

        Args:
            dataloader: DataLoader for captioning dataset
            output_file: Optional file to save predictions

        Returns:
            Dict with metric scores
        """
        # Collect predictions and references
        predictions = {}
        references = {}
        generated_token_counts = []
        clean_eos_count = 0

        for batch in tqdm(dataloader, desc="Generating captions"):
            if batch is None:
                continue

            # Move to device
            images = batch["image"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # Generate captions
            generated_ids = self.model.generate(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

            # Decode predictions
            for i in range(len(generated_ids)):
                prompt_len = input_ids[i].shape[0]
                # When generating from `inputs_embeds` (multimodal), HF may return only new tokens.
                seq = generated_ids[i]
                generated = seq[prompt_len:] if seq.shape[0] > prompt_len else seq
                generated_token_counts.append(int(generated.shape[0]))
                if generated.numel() > 0 and generated[-1].item() == self.model.tokenizer.eos_token_id:
                    clean_eos_count += 1

                caption = self.model.tokenizer.decode(
                    generated,
                    skip_special_tokens=True,
                ).strip()

                image_id = batch["image_id"][i].item() if isinstance(
                    batch["image_id"][i], torch.Tensor
                ) else batch["image_id"][i]

                predictions[image_id] = [caption]
                references[image_id] = batch["captions"][i]

        avg_generated_tokens = (
            sum(generated_token_counts) / len(generated_token_counts)
            if generated_token_counts else 0.0
        )
        eos_rate = clean_eos_count / len(generated_token_counts) if generated_token_counts else 0.0

        if not predictions:
            metrics = {"num_samples": 0}
        else:
            # Compute metrics
            metrics = self._compute_metrics(predictions, references)

        metrics["num_samples"] = len(predictions)
        metrics["avg_generated_tokens"] = avg_generated_tokens
        metrics["eos_rate"] = eos_rate

        # Save predictions
        if output_file:
            output_data = [
                {"image_id": img_id, "caption": caps[0]}
                for img_id, caps in predictions.items()
            ]
            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2)

        return metrics

    def _compute_metrics(
        self,
        predictions: Dict[int, List[str]],
        references: Dict[int, List[str]],
    ) -> Dict[str, float]:
        """
        Compute captioning metrics using pycocoevalcap.

        Falls back to simple BLEU if pycocoevalcap not available.
        """
        try:
            from pycocoevalcap.eval import COCOEvalCap
            from pycocotools.coco import COCO

            # Create COCO-format annotations
            coco_res = []
            coco_gt = {"images": [], "annotations": []}
            ann_id = 0

            for img_id in predictions:
                coco_gt["images"].append({"id": img_id})

                # Add predictions
                for cap in predictions[img_id]:
                    coco_res.append({
                        "image_id": img_id,
                        "caption": cap,
                    })

                # Add references
                for cap in references[img_id]:
                    coco_gt["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "caption": cap,
                    })
                    ann_id += 1

            # Create temporary files
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(coco_gt, f)
                gt_file = f.name

            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(coco_res, f)
                res_file = f.name

            # Run evaluation
            coco = COCO(gt_file)
            coco_result = coco.loadRes(res_file)

            coco_eval = COCOEvalCap(coco, coco_result)
            coco_eval.evaluate()

            # Clean up
            os.unlink(gt_file)
            os.unlink(res_file)

            return coco_eval.eval

        except ImportError:
            print("pycocoevalcap not installed, using simple BLEU")
            return self._compute_simple_bleu(predictions, references)

    def _compute_simple_bleu(
        self,
        predictions: Dict[int, List[str]],
        references: Dict[int, List[str]],
    ) -> Dict[str, float]:
        """
        Compute simple BLEU score as fallback.
        """
        if not predictions:
            return {"BLEU-1": 0, "BLEU-4": 0}

        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        except ImportError:
            print("NLTK not installed, returning zeros")
            return {"BLEU-1": 0, "BLEU-4": 0}

        smooth = SmoothingFunction().method1

        bleu1_scores = []
        bleu4_scores = []

        for img_id in predictions:
            pred = predictions[img_id][0].split()
            refs = [ref.split() for ref in references[img_id]]

            bleu1 = sentence_bleu(refs, pred, weights=(1, 0, 0, 0), smoothing_function=smooth)
            bleu4 = sentence_bleu(refs, pred, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

            bleu1_scores.append(bleu1)
            bleu4_scores.append(bleu4)

        return {
            "BLEU-1": sum(bleu1_scores) / len(bleu1_scores) * 100,
            "BLEU-4": sum(bleu4_scores) / len(bleu4_scores) * 100,
        }


def caption_collate_fn(batch: List[Optional[Dict[str, Any]]], pad_token_id: int) -> Optional[Dict[str, Any]]:
    """
    Custom collate function for captioning evaluation with variable-length sequences.

    Args:
        batch: List of sample dicts from COCOCaptionDataset
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
        "image_id": [item["image_id"] for item in batch],
        "captions": [item["captions"] for item in batch],
    }


def evaluate_coco_captioning(
    model,
    annotations_file: str,
    image_dir: str,
    output_file: Optional[str] = None,
    batch_size: int = 16,
) -> Dict[str, float]:
    """
    Convenience function to evaluate on COCO captioning.

    Args:
        model: AnyMAL model
        annotations_file: Path to COCO captions annotations
        image_dir: Path to COCO images
        output_file: Optional output file for predictions
        batch_size: Evaluation batch size

    Returns:
        Evaluation metrics
    """
    from data import get_vision_transform

    is_v2 = getattr(model, "architecture", "") == "anymal_v2"
    vision_type = getattr(model, "vision_encoder_type", "clip")
    vision_model = getattr(getattr(model, "image_encoder", None), "model_name", None)
    transform = get_vision_transform(
        vision_encoder_type=vision_type,
        vision_model_name=vision_model,
        image_size=384 if is_v2 else 224,
        is_train=False,
        use_augmentation=False,
    )

    dataset = COCOCaptionDataset(
        annotations_file=annotations_file,
        image_dir=image_dir,
        transform=transform,
        tokenizer=model.tokenizer,
        image_placeholder_token_id=getattr(model, "image_placeholder_token_id", None),
        num_image_tokens=getattr(model, "num_image_tokens", 0) if is_v2 else 0,
    )

    pad_token_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda b: caption_collate_fn(b, pad_token_id),
    )

    evaluator = CaptioningEvaluator(model)
    results = evaluator.evaluate(dataloader, output_file)

    print("\nCOCO Captioning Results:")
    for metric, score in results.items():
        print(f"  {metric}: {score:.2f}")

    return results
