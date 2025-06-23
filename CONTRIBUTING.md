# Contributing to MultiSpecVision

Thank you for your interest in contributing to MultiSpecVision! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MultiSpecVision
   ```

2. **Set up environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run tests**
   ```bash
   python test_setup.py
   python test_inference.py
   ```

## How to Contribute

### 🐛 Reporting Bugs
- Search existing issues first
- Use the bug report template
- Include system information and error logs
- Provide minimal reproduction steps

### 💡 Suggesting Features
- Check if the feature already exists or is planned
- Explain the use case and benefits
- Consider implementation complexity

### 🔧 Code Contributions

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

4. **Test your changes**
   ```bash
   python test_setup.py
   # Test specific functionality
   ```

5. **Commit your changes**
   ```bash
   git commit -m "Add: brief description of changes"
   ```

6. **Push and create a Pull Request**

## Code Style Guidelines

- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and small
- Comment complex logic

### File Naming Convention
- Use `multispec_` prefix for new model files
- Use descriptive names for utilities
- Follow existing patterns

## Testing

- All new features must include tests
- Ensure existing tests still pass
- Test both single-channel and multi-channel functionality
- Include edge cases in tests

## Documentation

- Update README if adding new features
- Add inline comments for complex code
- Update docstrings for API changes
- Include usage examples

## Review Process

1. All contributions require code review
2. Tests must pass
3. Documentation must be updated
4. Code style must be consistent

## Getting Help

- Open a discussion for questions
- Check existing documentation
- Contact maintainers for major changes

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
