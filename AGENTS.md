# Instructions for AI agents

This project follows ai-working-practices. Read its README at the start of every conversation, and check its CHANGELOG.md at least once a day in longer work.

Practices used (versions and changelog entry are recorded in `TASKS.md`):
- Task board: `TASKS.md`, with the plan in `PLAN.md` and decisions in `DECISIONS.md`.
- Risk list: `docs/risks.md`. Used because this is research that may not work.
- Fresh-eyes review: used before any write-up leaves the project, and before a result is treated as the foundation for the scale-up phase.
- Style guide: applied to `README.md` and any write-up. Working files follow their own conventions.

## The project

A research project testing whether ideas from Michael Levin's work on bioelectric pattern control (competent parts steered through a small set of goals) give useful AI methods. Start with `README.md` for what has been found, then `TASKS.md` for what is in progress.

## Working rules

- Experiments run on a GPU in Google Colab, in the owner's notebook. Each run costs a little of the owner's prepaid Colab compute units: disconnect the runtime (Runtime → Disconnect and delete runtime) as soon as a run finishes.
- Each script is self-contained and runs with `python <script> --quick` as a smoke test on CPU before any GPU run.
- Results are reported as written, including failures. A result that needed a fix is reported with the fix and the run before it.
- Every comparison between methods uses the same trained network, data and budget for each method, and pass criteria are fixed before running where possible.
