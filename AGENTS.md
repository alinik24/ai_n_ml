# Agent workflow

GitHub (`alinik24/ai_n_ml`) is canonical; no permanent local checkout is expected. Use `repo-dev ai_n_ml` or clone the current default branch, then verify origin, branch, HEAD, and working-tree state.

Run `./bootstrap.ps1`, the repository doctor, and its documented validation before bounded work. If no formal test suite covers the task, use the real import/notebook or example validation described by the project. Keep datasets, models, credentials, and generated outputs outside Git unless they are explicitly safe fixtures.

After changes, run affected validation, commit, push, verify the remote commit, and run `repo-release ai_n_ml` when inactive. Never use destructive dependency force-upgrades blindly.
