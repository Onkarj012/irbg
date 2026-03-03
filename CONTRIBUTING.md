# Contributing to IRBG

## Workflow
1. Create an Issue describing the work.
2. Create a branch from `main`:
   - `feat/<topic>`, `fix/<bug>`, `docs/<doc>`, `chore/<maintenance>`
3. Make small commits with clear messages.
4. Open a Pull Request (PR).
5. PR must pass CI checks and be reviewed before merging.

## Commit message style
We use Conventional Commits:
- `feat: ...` new feature
- `fix: ...` bug fix
- `docs: ...` documentation
- `chore: ...` tooling / setup
- `refactor: ...` code change without feature/bug fix

Examples:
- `feat: add openrouter provider wrapper`
- `fix: handle rate limit retry logic`
- `docs: explain scenario schema`

## Code quality
- Format/lint with `ruff`
- Add/update tests for core logic
- Do not commit `.env` or API keys