"""Benchmark harness for Insight's trust-critical claims.

Each benchmark writes machine-readable records (JSONL), a summary (JSON), and a
manifest that pins the code, environment, configuration, and artifact hashes.
The generated ``BENCHMARKS.md`` is a human-readable view of the same data.

Nothing in this package is imported by the shipped application; it is evidence
tooling only.
"""

__all__ = ["__version__"]

__version__ = "1.0"
