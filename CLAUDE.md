# Project Rules

## Environment

- **Conda environment**: `env6` (Python 3.10.18)
- **Conda binary**: `/home/jupiter/miniconda3/bin/conda`
- **Run commands**: always use `conda run -n env6` prefix, e.g. `conda run -n env6 python main.py`
- **Install packages**: `conda run -n env6 pip install <package>` — NEVER install globally (bare `pip install`)
- **Running tests**: `conda run -n env6 pytest` (parallel by default, 4 workers via pytest-xdist)
- **Run tests sequentially** (debug): `conda run -n env6 pytest -n 0`
- **Custom worker count**: `conda run -n env6 pytest -n 2`
- **Do NOT** use bare `python`, `pip`, or `conda` without the full path prefix — they resolve to the system Python, not the project environment.

## File size discipline

- Hard limit: **800 lines** per Python file. If a change would push a file over 800 lines, split it first.
- Split by **domain/responsibility**, not mechanically. Group related logic together so each module has a clear, coherent purpose.
- When a file exceeds 800 lines, refactor it into smaller modules under a package (directory with `__init__.py`) or into separate service files.
- Adapters, services, models, and utilities should each live in their own appropriate directory under `app/`.
