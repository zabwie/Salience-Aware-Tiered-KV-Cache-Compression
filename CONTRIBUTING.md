# Contributing to TTKV

Thank you for your interest in contributing to TTKV! This document provides guidelines for contributing to the project.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/zabwie/ttkv.git
   cd ttkv
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev,viz]"
   ```

## Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting (line length: 100)
- **Flake8**: Linting
- **mypy**: Type checking

Run all checks before committing:

```bash
black src/ttkv
flake8 src/ttkv
mypy src/ttkv
```

## Testing

Run the test suite:

```bash
# Unit tests
pytest tests/core/ -v

# Integration tests
pytest tests/integration/ -v

# All tests with coverage
pytest tests/ --cov=ttkv --cov-report=html
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

## Reporting Issues

When reporting issues, please include:

- Python version
- PyTorch version
- Operating system
- Steps to reproduce
- Expected behavior
- Actual behavior

## License

By contributing to TTKV, you agree that your contributions will be licensed under the MIT License.
