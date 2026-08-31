# Contributing

## Change workflow

Keep changes small enough that scientific behavior and performance consequences
can be reviewed independently. Each implementation step should normally include:

1. a reference result or invariant derived from the pinned Python oracle;
2. focused unit or integration tests;
3. formatting, Clippy, tests, and documentation checks; and
4. benchmark evidence when the change makes a performance claim.

Do not commit FVCOM datasets, the local Python UTide checkout, project-local
toolchains, profiler captures, or raw benchmark runs.

## Required checks

```console
cargo fmt --all -- --check
cargo ci
cargo test-all
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps --locked
uv run --project benchmarks/python --locked ruff format --check benchmarks/python
uv run --project benchmarks/python --locked ruff check benchmarks/python
uv run --project benchmarks/python --locked \
  python -m unittest discover -s benchmarks/python/tests -v
```

Run release-mode correctness tests as well when floating-point behavior could be
affected by optimization.

## Scientific changes

Document equations, units, conventions, and the relevant Python UTide behavior
next to the implementation. Compare phases with circular distance and match
constituents by identity before comparing arrays. Never weaken a tolerance merely
to make a failing case pass; first explain the numerical difference and its
scientific consequence.

## Performance changes

Do not infer application speed from a microbenchmark alone. Retain the unoptimized
comparison, report all repetitions, and distinguish solve-only time from NetCDF
I/O and result serialization. Follow `BENCHMARK_PLAN.md` for the full protocol.
