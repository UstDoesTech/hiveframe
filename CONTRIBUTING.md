# Contributing to HiveFrame

Thank you for your interest in contributing to HiveFrame! This document provides guidelines for contributing.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/hiveframe/hiveframe.git
   cd hiveframe
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

4. Run tests:
   ```bash
   pytest
   ```

## Code Style

We use the following tools for code quality:

- **black** for code formatting
- **ruff** for linting
- **mypy** for type checking

Run all checks:
```bash
black src tests
ruff check src tests
mypy src
```

## Testing

- Write tests for all new functionality
- Tests go in the `tests/` directory
- Use pytest fixtures for shared setup
- Mark slow tests with `@pytest.mark.slow`

Run tests:
```bash
pytest                    # All tests
pytest -v                 # Verbose output
pytest -m "not slow"      # Skip slow tests
pytest tests/test_core.py # Specific file
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit with descriptive messages
6. Push to your fork
7. Open a Pull Request

### PR Checklist

- [ ] Tests pass (`pytest`)
- [ ] Code is formatted (`black`)
- [ ] Linting passes (`ruff check`)
- [ ] Type hints added where appropriate
- [ ] Documentation updated if needed
- [ ] CHANGELOG.md updated for notable changes

## Bee Colony Patterns

When contributing, keep the biomimicry metaphor in mind:

- **Waggle Dance**: Communication of quality/success
- **Employed/Onlooker/Scout**: Different worker behaviors
- **Pheromones**: Indirect coordination signals
- **Colony Temperature**: System load awareness
- **Abandonment**: Self-healing from stuck states

## Reporting Issues

When reporting bugs, please include:

- Python version
- HiveFrame version
- Operating system
- Minimal reproducible example
- Expected vs actual behavior

## Feature Requests

We welcome feature requests! Please describe:

- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
