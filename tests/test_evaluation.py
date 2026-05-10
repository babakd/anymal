"""
Focused tests for evaluation dataset and generation-health surfaces.
"""

import json

import torch
from PIL import Image


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    eot_token_id = 3

    def __call__(
        self,
        text,
        return_tensors=None,
        padding=False,
        truncation=True,
        max_length=64,
    ):
        token_ids = [10, 11, 12]
        return {
            "input_ids": torch.tensor([token_ids], dtype=torch.long),
            "attention_mask": torch.ones(1, len(token_ids), dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        ids = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else list(token_ids)
        if skip_special_tokens:
            ids = [
                token_id for token_id in ids
                if token_id not in {self.pad_token_id, self.eos_token_id, self.eot_token_id}
            ]
        return " ".join(f"tok{token_id}" for token_id in ids)

    def get_vocab(self):
        return {"<|eot_id|>": self.eot_token_id}


def _write_coco_caption_fixture(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(image_dir / "present.jpg")

    annotations = {
        "images": [
            {"id": 1, "file_name": "present.jpg"},
            {"id": 2, "file_name": "missing.jpg"},
        ],
        "annotations": [
            {"image_id": 1, "caption": "a red square"},
            {"image_id": 2, "caption": "a missing image"},
        ],
    }
    annotations_file = tmp_path / "captions.json"
    annotations_file.write_text(json.dumps(annotations))
    return annotations_file, image_dir


def test_coco_caption_dataset_filters_missing_images_and_inserts_v2_placeholders(tmp_path):
    from evaluation.captioning_eval import COCOCaptionDataset

    annotations_file, image_dir = _write_coco_caption_fixture(tmp_path)
    dataset = COCOCaptionDataset(
        annotations_file=str(annotations_file),
        image_dir=str(image_dir),
        transform=lambda image: torch.zeros(3, 4, 4),
        tokenizer=TinyTokenizer(),
        image_placeholder_token_id=99,
        num_image_tokens=4,
    )

    assert len(dataset) == 1

    sample = dataset[0]
    assert sample is not None
    assert sample["image_id"] == 1
    assert sample["input_ids"][:4].tolist() == [99, 99, 99, 99]
    assert sample["input_ids"][4:].tolist() == [10, 11, 12]
    assert sample["attention_mask"].tolist() == [1, 1, 1, 1, 1, 1, 1]


def test_caption_collate_fn_filters_invalid_samples():
    from evaluation.captioning_eval import caption_collate_fn

    batch = [
        None,
        {
            "image": torch.zeros(3, 4, 4),
            "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "attention_mask": torch.ones(3, dtype=torch.long),
            "image_id": 1,
            "captions": ["one"],
        },
        {
            "image": torch.ones(3, 4, 4),
            "input_ids": torch.tensor([4, 5], dtype=torch.long),
            "attention_mask": torch.ones(2, dtype=torch.long),
            "image_id": 2,
            "captions": ["two"],
        },
    ]

    collated = caption_collate_fn(batch, pad_token_id=0)

    assert collated is not None
    assert collated["image"].shape == (2, 3, 4, 4)
    assert collated["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]]
    assert collated["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert caption_collate_fn([None], pad_token_id=0) is None


def test_captioning_evaluator_reports_generation_health_metrics():
    from evaluation.captioning_eval import CaptioningEvaluator

    class TinyModel:
        tokenizer = TinyTokenizer()

        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, images, input_ids, attention_mask, max_new_tokens, do_sample):
            new_tokens = torch.tensor(
                [
                    [50, self.tokenizer.eos_token_id],
                    [51, 52],
                ],
                dtype=torch.long,
                device=input_ids.device,
            )
            return torch.cat([input_ids, new_tokens], dim=1)

    batch = {
        "image": torch.zeros(2, 3, 4, 4),
        "input_ids": torch.tensor([[10, 11, 12], [10, 11, 12]], dtype=torch.long),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "image_id": [1, 2],
        "captions": [["caption one"], ["caption two"]],
    }

    evaluator = CaptioningEvaluator(TinyModel(), device=torch.device("cpu"), max_new_tokens=2)
    evaluator._compute_metrics = lambda predictions, references: {"BLEU-1": 12.0}

    results = evaluator.evaluate([batch, None])

    assert results["BLEU-1"] == 12.0
    assert results["num_samples"] == 2
    assert results["avg_generated_tokens"] == 2.0
    assert results["eos_rate"] == 0.5


def test_vqa_evaluator_counts_first_stop_token_not_padding():
    from evaluation.vqa_eval import VQAEvaluator

    class TinyModel:
        tokenizer = TinyTokenizer()

        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, images, input_ids, attention_mask, max_new_tokens, do_sample):
            new_tokens = torch.tensor(
                [
                    [50, self.tokenizer.eot_token_id, self.tokenizer.pad_token_id, self.tokenizer.pad_token_id],
                    [51, 52, 53, 54],
                ],
                dtype=torch.long,
                device=input_ids.device,
            )
            return torch.cat([input_ids, new_tokens], dim=1)

    batch = {
        "image": torch.zeros(2, 3, 4, 4),
        "input_ids": torch.tensor([[10, 11], [10, 11]], dtype=torch.long),
        "attention_mask": torch.ones(2, 2, dtype=torch.long),
        "image_id": [101, 102],
        "question_id": [1, 2],
        "answers": [["tok50"], ["tok51"]],
        "answer_type": ["other", "other"],
    }

    evaluator = VQAEvaluator(TinyModel(), device=torch.device("cpu"), max_new_tokens=4)
    results = evaluator.evaluate([batch])

    assert results["avg_generated_tokens"] == 3.0
    assert results["eos_rate"] == 0.5
    assert results["hit_max_new_tokens_rate"] == 0.5


def test_vqa_collate_fn_left_pads_for_decoder_only_generation():
    from evaluation.vqa_eval import vqa_collate_fn

    batch = [
        {
            "image": torch.zeros(3, 4, 4),
            "input_ids": torch.tensor([10, 11, 12], dtype=torch.long),
            "attention_mask": torch.ones(3, dtype=torch.long),
            "image_id": 101,
            "question_id": 1,
            "question": "long?",
            "answers": ["yes"],
            "answer_type": "yes/no",
            "question_type": "",
        },
        {
            "image": torch.ones(3, 4, 4),
            "input_ids": torch.tensor([20], dtype=torch.long),
            "attention_mask": torch.ones(1, dtype=torch.long),
            "image_id": 102,
            "source_image_id": 201,
            "image_control": "shuffled_image",
            "question_id": 2,
            "question": "short?",
            "answers": ["no"],
            "answer_type": "yes/no",
            "question_type": "",
        },
    ]

    collated = vqa_collate_fn(batch, pad_token_id=0)

    assert collated["input_ids"].tolist() == [[10, 11, 12], [0, 0, 20]]
    assert collated["attention_mask"].tolist() == [[1, 1, 1], [0, 0, 1]]
    assert collated["source_image_id"] == [101, 201]
    assert collated["image_control"] == ["none", "shuffled_image"]


def test_vqa_evaluator_reports_strict_accuracy_and_role_prefix_rate():
    from evaluation.vqa_eval import VQAEvaluator

    class PrefixTokenizer(TinyTokenizer):
        def decode(self, token_ids, skip_special_tokens=True):
            ids = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else list(token_ids)
            if skip_special_tokens:
                ids = [
                    token_id for token_id in ids
                    if token_id not in {self.pad_token_id, self.eos_token_id, self.eot_token_id}
                ]
            if ids == [70, 71, 72]:
                return "assistant\n\nyes"
            if ids == [73]:
                return "no"
            return super().decode(ids, skip_special_tokens=skip_special_tokens)

    class TinyModel:
        tokenizer = PrefixTokenizer()

        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, images, input_ids, attention_mask, max_new_tokens, do_sample):
            new_tokens = torch.tensor(
                [
                    [70, 71, 72, self.tokenizer.eot_token_id],
                    [73, self.tokenizer.eot_token_id, self.tokenizer.pad_token_id, self.tokenizer.pad_token_id],
                ],
                dtype=torch.long,
                device=input_ids.device,
            )
            return torch.cat([input_ids, new_tokens], dim=1)

    batch = {
        "image": torch.zeros(2, 3, 4, 4),
        "input_ids": torch.tensor([[10, 11], [10, 11]], dtype=torch.long),
        "attention_mask": torch.ones(2, 2, dtype=torch.long),
        "image_id": [101, 102],
        "question_id": [1, 2],
        "answers": [["yes", "yes", "yes"], ["no", "no", "no"]],
        "answer_type": ["yes/no", "yes/no"],
    }

    evaluator = VQAEvaluator(TinyModel(), device=torch.device("cpu"), max_new_tokens=4)
    results = evaluator.evaluate([batch])

    assert results["accuracy"] == 100.0
    assert results["strict_accuracy"] == 50.0
    assert results["assistant_role_prefix_rate"] == 0.5
    assert results["assistant_role_prefix_rate_yes_no"] == 0.5
