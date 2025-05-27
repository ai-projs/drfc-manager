# Contributing to drfc-manager

Thank you for your interest in contributing to **drfc-manager**! 🎉  
We welcome all kinds of contributions: bug reports, feature requests, code, documentation, and more.

## Getting Started

1. **Fork the repository** and clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/drfc-manager.git
   cd drfc-manager
   ```

2. **Set up your environment**  
   We recommend using [Poetry](https://python-poetry.org/) for dependency management:
   ```bash
   poetry install
   ```

3. **Install pre-commit hooks**  
   This ensures code style and formatting checks run before each commit:
   ```bash
   pre-commit install
   ```

## Making Changes

- Create a new branch for your work:
  ```bash
  git checkout -b my-feature-branch
  ```

- Make your changes and commit them.  
  Pre-commit will run checks automatically.
<!-- 
- Run tests to make sure everything works:
  ```bash
  poetry run pytest
  ``` -->

## Linting and Formatting

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting.  
Ruff will run automatically via pre-commit hooks, but you can also run it manually:
```bash
ruff check .
run ruff format .
```

## Submitting a Pull Request

1. Push your branch to your fork:
   ```bash
   git push origin my-feature-branch
   ```

2. Open a pull request (PR) against the `dev` branch of the main repository.

3. Fill out the PR template and describe your changes.

## Code Review & Merging

- All PRs are reviewed by maintainers.
- Please respond to feedback and make requested changes.
- Once approved, your PR will be merged!

## Additional Notes

- For major changes, please open an issue first to discuss what you'd like to change.
<!-- - Please make sure to update tests as appropriate. -->
- See the [README](README.md) for more information about the project.

---

Thank you for helping make **drfc-manager** better! 🚗💨
