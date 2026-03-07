import modal_train


def test_resume_checkpoint_skips_stage1_autodiscovery(monkeypatch):
    called = {"value": False}

    def fail_if_called(_root_dir):
        called["value"] = True
        raise AssertionError("auto-discovery should be skipped while resuming")

    monkeypatch.setattr(modal_train, "_collect_checkpoint_candidates", fail_if_called)

    checkpoint, source = modal_train._resolve_finetune_pretrain_checkpoint(
        pretrain_checkpoint=None,
        resume_checkpoint="/checkpoints/finetune-output/run-0002/checkpoint-100",
    )

    assert checkpoint is None
    assert source == "resume"
    assert called["value"] is False


def test_auto_discovers_latest_stage1_checkpoint(monkeypatch):
    monkeypatch.setattr(
        modal_train,
        "_collect_checkpoint_candidates",
        lambda _root_dir: [
            (100.0, 100, "/checkpoints/pretrain-output/checkpoint-100"),
            (200.0, 200, "/checkpoints/pretrain-output/checkpoint-200"),
        ],
    )

    checkpoint, source = modal_train._resolve_finetune_pretrain_checkpoint(
        pretrain_checkpoint=None,
        resume_checkpoint=None,
    )

    assert checkpoint == "/checkpoints/pretrain-output/checkpoint-200"
    assert source == "auto"
