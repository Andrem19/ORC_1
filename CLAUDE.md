# Project Rules

## File size discipline

- Hard limit: **800 lines** per Python file. If a change would push a file over 800 lines, split it first.
- Split by **domain/responsibility**, not mechanically. Group related logic together so each module has a clear, coherent purpose.
- When a file exceeds 800 lines, refactor it into smaller modules under a package (directory with `__init__.py`) or into separate service files.
- Adapters, services, models, and utilities should each live in their own appropriate directory under `app/`.
