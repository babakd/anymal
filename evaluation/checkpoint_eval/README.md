# Checkpoint Evaluators

The checkpoint evaluator implementations live here:

- `vqa_checkpoint_eval.py`
- `gqa_checkpoint_eval.py`
- `pope_checkpoint_eval.py`

The old root entrypoints remain as compatibility wrappers, so existing Modal
commands still work:

```bash
modal run vqa_checkpoint_eval.py --help
modal run gqa_checkpoint_eval.py --help
modal run pope_checkpoint_eval.py --help
```

