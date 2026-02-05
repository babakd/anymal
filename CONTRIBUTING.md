# Contributing to AnyMAL

Thank you for your interest in contributing to AnyMAL! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/anymal.git`
3. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Install development dependencies: `pip install pytest black isort`

## Development Workflow

### Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting

Before submitting a PR, run:
```bash
black .
isort .
```

### Running Tests

```bash
pytest tests/test_model.py -v
```

### Making Changes

1. Create a new branch for your feature: `git checkout -b feature/my-feature`
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Run code formatters
6. Commit your changes with a clear message
7. Push to your fork and submit a pull request

## Pull Request Guidelines

- Keep PRs focused on a single feature or fix
- Include a clear description of the changes
- Reference any related issues
- Update documentation if needed
- Add tests for new functionality

## Reporting Issues

When reporting issues, please include:

- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (Python version, PyTorch version, GPU, etc.)
- Relevant error messages or logs

## Questions?

Feel free to open an issue for any questions about contributing.
