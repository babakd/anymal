"""
Modal inference script for architecture-aware checkpoint progression runs.
"""

import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path

import modal


app = modal.App("anymal-inference")

volume = modal.Volume.from_name("anymal-checkpoints", create_if_missing=True)

PROJECT_DIR = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "open_clip_torch>=2.23.0",
        "pillow>=10.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "sentencepiece>=0.1.99",
        "huggingface_hub>=0.19.0",
    )
    .add_local_dir(PROJECT_DIR, remote_path="/root/anymal", copy=False)
)


@app.cls(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/checkpoints": volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
class Inference:
    @modal.enter()
    def setup(self):
        import sys

        sys.path.insert(0, "/root/anymal")

        self.llama_path = "/checkpoints/llama3-8b-instruct"
        if not os.path.exists(os.path.join(self.llama_path, "config.json")):
            raise RuntimeError(
                "LLaMA weights not found. Run training first to cache weights."
            )

        json_path = "/checkpoints/llava_data/llava_instruct_150k.json"
        if not os.path.exists(json_path):
            raise RuntimeError(
                "LLaVA dataset JSON not found. Run training first to cache data."
            )

        image_dir = "/checkpoints/coco_images"
        if not os.path.exists(image_dir):
            raise RuntimeError(
                "COCO images not found. Run training first to cache images."
            )

    @staticmethod
    def _parse_checkpoint_step(name: str):
        if not str(name).startswith("checkpoint-"):
            return None
        try:
            return int(str(name).split("-", 1)[1])
        except (IndexError, ValueError):
            return None

    def _iter_checkpoint_dirs(self, root_dir: str):
        for current_root, dirnames, _filenames in os.walk(root_dir):
            base = os.path.basename(current_root.rstrip("/"))
            if self._parse_checkpoint_step(base) is not None:
                yield current_root
                dirnames[:] = []

    def _discover_checkpoints(self):
        """Find saved checkpoints grouped by architecture."""
        from model_metadata import read_model_metadata, resolve_checkpoint_architecture

        grouped = {}
        latest_pretrain_mtime = {}

        for stage, root_dir in (
            ("pretrain", "/checkpoints/pretrain-output"),
            ("finetune", "/checkpoints/finetune-output"),
        ):
            if not os.path.exists(root_dir):
                continue

            for ckpt_path in self._iter_checkpoint_dirs(root_dir):
                step = self._parse_checkpoint_step(os.path.basename(ckpt_path))
                projector_path = os.path.join(ckpt_path, "projector.pt")
                if step is None or not os.path.exists(projector_path):
                    continue

                architecture, _ = resolve_checkpoint_architecture(ckpt_path)
                metadata = read_model_metadata(ckpt_path) or {"architecture": architecture}
                metadata["architecture"] = architecture
                mtime = os.path.getmtime(projector_path)

                grouped.setdefault(architecture, []).append(
                    {
                        "step": step,
                        "checkpoint_path": ckpt_path,
                        "stage": stage,
                        "metadata": metadata,
                        "mtime": mtime,
                    }
                )

                if stage == "pretrain":
                    latest_pretrain_mtime[architecture] = max(
                        latest_pretrain_mtime.get(architecture, 0),
                        mtime,
                    )

        for architecture, runs in grouped.items():
            filtered = []
            pretrain_cutoff = latest_pretrain_mtime.get(architecture, 0)
            for run in runs:
                if run["stage"] == "finetune" and pretrain_cutoff and run["mtime"] < pretrain_cutoff:
                    print(
                        f"Skipping stale {architecture} finetune checkpoint "
                        f"{run['checkpoint_path']}"
                    )
                    continue
                filtered.append(run)
            grouped[architecture] = sorted(filtered, key=lambda item: (item["step"], item["stage"]))

        print(
            "Discovered checkpoints:",
            {
                arch: [(item["step"], item["stage"]) for item in items]
                for arch, items in grouped.items()
            },
        )
        return grouped

    def _select_val_examples(self, num_examples=30):
        from data.instruction_dataset import InstructionDataset
        from data.dataset_splitter import deterministic_train_val_split
        from transformers import AutoTokenizer

        json_path = "/checkpoints/llava_data/llava_instruct_150k.json"
        image_dir = "/checkpoints/coco_images"

        tokenizer = AutoTokenizer.from_pretrained(self.llama_path)
        dataset = InstructionDataset(
            data_path=json_path,
            image_dir=image_dir,
            tokenizer=tokenizer,
            image_size=224,
            max_length=512,
            filter_to_available_images=True,
        )

        _, val_dataset = deterministic_train_val_split(dataset, val_fraction=0.05)
        val_size = len(val_dataset)
        if num_examples >= val_size:
            selected_indices = list(range(val_size))
        else:
            step = val_size / num_examples
            selected_indices = [int(i * step) for i in range(num_examples)]

        examples = []
        for idx in selected_indices:
            original_idx = val_dataset.indices[idx]
            examples.append(dataset.samples[original_idx])
        return examples

    @staticmethod
    def _extract_question_and_answer(sample):
        conversations = sample.get("conversations", [])
        question = ""
        answer = ""
        for turn in conversations:
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))
            content = content.replace("<image>", "").replace("\n", " ").strip()
            if role in ("human", "user") and not question:
                question = content
            elif role in ("gpt", "assistant") and not answer:
                answer = content
        return question, answer

    @staticmethod
    def _image_to_base64(image_path, max_size=384):
        from PIL import Image

        try:
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((max_size, max_size))
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as exc:
            print(f"Warning: could not encode image {image_path}: {exc}")
            return None

    def _get_model_kwargs(self, architecture: str, metadata: dict, use_qlora: bool):
        projector_type = metadata.get("projector_type", "PerceiverResampler")
        projector_type = {
            "PerceiverResampler": "perceiver",
            "LinearProjector": "linear",
        }.get(projector_type, projector_type)

        common = {
            "llm_model_name": self.llama_path,
            "use_qlora": use_qlora,
            "lora_r": 64 if use_qlora else 0,
            "lora_alpha": 16 if use_qlora else 0,
            "gradient_checkpointing": False,
            "use_flash_attention": False,
        }
        if architecture == "anymal_v1":
            common.update(
                {
                    "vision_model_name": metadata.get("vision_model_name", "ViT-L-14"),
                    "vision_pretrained": metadata.get("vision_pretrained", "openai"),
                    "projector_type": projector_type,
                    "num_image_tokens": metadata.get("num_image_tokens", 64),
                }
            )
        elif architecture == "anymal_v2":
            common.update(
                {
                    "vision_encoder_type": metadata.get("vision_encoder_type", "siglip2"),
                    "vision_model_name": metadata.get(
                        "vision_model_name",
                        "google/siglip2-so400m-patch14-384",
                    ),
                    "token_compressor_type": metadata.get(
                        "token_compressor_type",
                        "learned",
                    ),
                    "bottleneck_dim": metadata.get("bottleneck_dim", 2048),
                    "max_image_tokens": metadata.get("max_image_tokens", 256),
                    "min_image_tokens": metadata.get("min_image_tokens", metadata.get("max_image_tokens", 256)),
                }
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        return common

    def _load_base_model(self, architecture: str, metadata: dict, use_qlora: bool = True):
        import torch
        from models import create_model

        print(f"Loading {architecture} base model (use_qlora={use_qlora})...")
        model = create_model(
            architecture=architecture,
            **self._get_model_kwargs(architecture, metadata, use_qlora=use_qlora),
        )
        model.eval()
        model.cuda()
        torch.cuda.empty_cache()
        return model

    @staticmethod
    def _capture_visual_bridge_state(model):
        state = {}
        for label, module in model.get_visual_bridge_modules().items():
            state[label] = {
                name: tensor.detach().clone()
                for name, tensor in module.state_dict().items()
            }
        return state

    @staticmethod
    def _restore_visual_bridge_state(model, bridge_state):
        for label, module in model.get_visual_bridge_modules().items():
            if label in bridge_state:
                module.load_state_dict(bridge_state[label])

    @staticmethod
    def _load_visual_bridge_from_checkpoint(model, checkpoint_path: str):
        import torch

        first_module = next(iter(model.get_visual_bridge_modules().values()))
        device = next(first_module.parameters()).device

        projector_path = os.path.join(checkpoint_path, "projector.pt")
        if os.path.exists(projector_path):
            model.projector.load_state_dict(torch.load(projector_path, map_location=device))

        compressor_path = os.path.join(checkpoint_path, "token_compressor.pt")
        if hasattr(model, "token_compressor") and os.path.exists(compressor_path):
            model.token_compressor.load_state_dict(torch.load(compressor_path, map_location=device))

    @staticmethod
    def _swap_lora(model, checkpoint_path: str):
        from peft import set_peft_model_state_dict

        llm_path = os.path.join(checkpoint_path, "llm")
        if not os.path.exists(llm_path):
            print(f"Warning: no LoRA weights at {llm_path}")
            return

        adapter_file = os.path.join(llm_path, "adapter_model.safetensors")
        if os.path.exists(adapter_file):
            from safetensors.torch import load_file

            adapter_state = load_file(adapter_file)
        else:
            import torch

            adapter_file = os.path.join(llm_path, "adapter_model.bin")
            adapter_state = torch.load(adapter_file, map_location="cpu")
        set_peft_model_state_dict(model.llm.model, adapter_state)

    @staticmethod
    def _reset_lora_weights(model):
        reset_count = 0
        for _name, module in model.llm.model.named_modules():
            if hasattr(module, "lora_B"):
                for _key, linear in module.lora_B.items():
                    linear.weight.data.zero_()
                    reset_count += 1
        print(f"Reset {reset_count} LoRA B matrices to zero")

    def _run_inference(self, model, examples):
        import torch
        from PIL import Image

        from data.data_utils import build_image_transform_from_model
        from data.multimodal_inputs import build_multimodal_chat_input

        transform = build_image_transform_from_model(
            model,
            is_train=False,
            use_augmentation=False,
        )
        image_dir = "/checkpoints/coco_images"

        predictions = []
        for idx, sample in enumerate(examples):
            image_filename = sample.get("image", "")
            image_path = os.path.join(image_dir, image_filename)
            question, _answer = self._extract_question_and_answer(sample)

            try:
                image = Image.open(image_path).convert("RGB")
                image_tensor = transform(image).unsqueeze(0).cuda()
                prompt = build_multimodal_chat_input(
                    tokenizer=model.tokenizer,
                    user_text=question,
                    image_placeholder_token_id=model.image_placeholder_token_id,
                    num_image_tokens=model.fixed_image_token_count,
                )
                input_ids = prompt["input_ids"].unsqueeze(0).cuda()
                attention_mask = prompt["attention_mask"].unsqueeze(0).cuda()

                with torch.no_grad():
                    output_ids = model.generate(
                        images=image_tensor,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=128,
                        do_sample=False,
                    )

                prompt_len = input_ids.shape[1]
                generated_ids = output_ids[0, prompt_len:]
                prediction = model.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                ).strip()
            except Exception as exc:
                prediction = f"[Error: {exc}]"
                print(f"Example {idx}: error - {exc}")

            predictions.append(prediction)
            if (idx + 1) % 5 == 0 or idx == 0:
                print(f"  Example {idx + 1}/{len(examples)}: {prediction[:80]}...")

        return predictions

    @modal.method()
    def run(self, num_examples: int = 30):
        import gc
        import torch

        start_time = time.time()
        checkpoints_by_arch = self._discover_checkpoints()
        if not checkpoints_by_arch:
            raise RuntimeError("No checkpoints found to evaluate.")

        examples = self._select_val_examples(num_examples)
        image_dir = "/checkpoints/coco_images"
        example_list = []
        for idx, sample in enumerate(examples):
            question, answer = self._extract_question_and_answer(sample)
            image_filename = sample.get("image", "")
            image_path = os.path.join(image_dir, image_filename)
            example_list.append(
                {
                    "idx": idx,
                    "image_b64": self._image_to_base64(image_path),
                    "image_filename": image_filename,
                    "question": question,
                    "ground_truth": answer,
                }
            )

        runs = []
        for architecture, checkpoints in sorted(checkpoints_by_arch.items()):
            print(f"\n{'=' * 60}\nRunning progression for {architecture}\n{'=' * 60}")
            model_metadata = checkpoints[0]["metadata"] if checkpoints else {"architecture": architecture}
            model = self._load_base_model(
                architecture=architecture,
                metadata=model_metadata,
                use_qlora=True,
            )
            initial_bridge_state = self._capture_visual_bridge_state(model)

            steps = []
            for checkpoint in ([{"step": 0, "checkpoint_path": None, "stage": "base", "metadata": model_metadata}] + checkpoints):
                step = checkpoint["step"]
                stage = checkpoint["stage"]
                checkpoint_path = checkpoint["checkpoint_path"]
                print(f"\n[{architecture}] step={step} stage={stage}")

                if checkpoint_path is not None:
                    self._load_visual_bridge_from_checkpoint(model, checkpoint_path)
                else:
                    self._restore_visual_bridge_state(model, initial_bridge_state)

                if stage == "finetune" and checkpoint_path is not None:
                    self._swap_lora(model, checkpoint_path)
                else:
                    self._reset_lora_weights(model)

                predictions = self._run_inference(model, examples)
                steps.append(
                    {
                        "architecture": architecture,
                        "step": step,
                        "stage": stage,
                        "checkpoint_path": checkpoint_path,
                        "predictions": [
                            {"idx": idx, "prediction": prediction}
                            for idx, prediction in enumerate(predictions)
                        ],
                    }
                )

            runs.append(
                {
                    "architecture": architecture,
                    "steps": steps,
                }
            )

            del model, initial_bridge_state
            gc.collect()
            torch.cuda.empty_cache()

        result = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "num_examples": len(example_list),
                "architectures": [run["architecture"] for run in runs],
                "elapsed_seconds": round(time.time() - start_time, 1),
            },
            "examples": example_list,
            "runs": runs,
        }

        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        output_path = f"/checkpoints/predictions_{ts}.json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle)
        volume.commit()
        print(f"Saved predictions to {output_path}")
        return result


@app.local_entrypoint()
def main(num_examples: int = 30):
    inference = Inference()
    result = inference.run.remote(num_examples=num_examples)

    ts = (
        result["metadata"]["timestamp"]
        .replace(":", "")
        .replace("-", "")
        .replace("T", "_")
        .replace("Z", "")
    )
    output_path = f"predictions_{ts}.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"Saved {output_path}")
    print(f"  architectures: {result['metadata']['architectures']}")
    print(f"  examples: {result['metadata']['num_examples']}")
    print(f"  elapsed: {result['metadata']['elapsed_seconds']}s")
