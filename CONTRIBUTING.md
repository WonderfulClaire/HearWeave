# Contributing

Contributions are welcome when they keep the toolkit reproducible and explicit about assumptions.

1. Open an issue for a new algorithm or geometry preset.
2. Include units, coordinate conventions, failure modes, and a synthetic test.
3. Run `ruff check .` and `python -m unittest discover -s tests -v`.
4. Update `docs/ALGORITHMS.md` and the changelog when behavior changes.

Do not add private recordings, third-party datasets without redistribution rights, or performance claims without a documented protocol.
