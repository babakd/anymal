# Contributing to AnyMAL

Thanks for taking a look at the project. This repo is a research codebase, so
small, focused improvements are easiest to review and easiest for future readers
to trust.

## Getting Started

1. Fork the repository.
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/anymal.git
   cd anymal
   ```
3. Create an environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   python3 -m pip install --upgrade pip
   python3 -m pip install -r requirements.txt
   ```
4. If you plan to run LLM-backed training or model tests, make sure you have
   Hugging Face access to `meta-llama/Meta-Llama-3-8B-Instruct`.

## Development Workflow

Create a branch for each focused change:

```bash
git checkout -b feature/my-change
```

Before opening a pull request, run the checks that match the scope of your
change:

```bash
black .
isort .
python3 -m pytest tests -q
```

For a narrower model/training change, these are useful starting points:

```bash
python3 -m pytest tests/test_model.py -q
python3 -m pytest tests/test_training.py -q
python3 -m pytest tests/test_evaluation.py -q
python3 -m pytest tests/test_health_monitor.py -q
```

Some tests import PyTorch and other heavy ML dependencies. If collection fails
because dependencies are not installed, mention that in the PR.

## Documentation Expectations

Please update public docs when changing:

- Setup or dependency requirements
- Config names, command-line flags, or dataset paths
- Model architecture behavior or checkpoint compatibility
- Training/evaluation workflows
- Modal usage

Keep research notes and result artifacts separate from user-facing setup docs
when possible. The README should stay oriented around what a new user can run.

## Pull Request Guidelines

- Keep PRs focused on one feature, fix, or experiment cleanup.
- Include a clear summary of what changed and why.
- Call out any required data, checkpoint, or secret that is not included in the
  repo.
- Add or update tests for behavior changes.
- Do not commit private tokens, downloaded model weights, large datasets, or
  generated checkpoint directories.

## Reporting Issues

When reporting a problem, include:

- The command you ran
- The config file and architecture variant
- Python, PyTorch, CUDA, and GPU details
- Whether the run is local or on Modal
- The relevant error message or log excerpt
- Any local dataset/checkpoint paths involved

## Questions

Open an issue with the context you have. Even partial reproduction details are
helpful.
