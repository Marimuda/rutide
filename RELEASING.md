# Release process

RUTide uses one version for the Rust workspace, command-line application, and
Python distribution. Cargo is the version source; the Python build reads the
same package version through maturin.

## Prepare

1. Before the first public release, configure the canonical Git remote and add
   matching repository/documentation/issues URLs to Cargo and `pyproject.toml`.
   Do not publish placeholder ownership metadata.
2. Update `CHANGELOG.md` and move completed entries out of `Unreleased`.
3. Set the workspace version in `Cargo.toml` and update exact internal dependency
   requirements.
4. Update `crates/rutide-cli/compatibility/feature-matrix-v1.json` and verify any
   NetCDF/JSON schema changes have incremented their independent schema constants.
5. Run the full Rust, Python, documentation, package, and wheel checks documented
   in the repository README. Inspect `cargo package --workspace --locked` and
   install both the wheel and source distribution into clean environments; a
   successful in-tree test run alone is not a release check.
6. Commit the release, create an annotated `vX.Y.Z` tag, and push both commit and
   tag. Signing is optional and must follow the maintainer's explicit release
   policy. Tags must point at a clean commit whose reported package versions
   agree.

## Build and publish

Release workflows build artifacts from the tag rather than from a mutable branch.
Publish in dependency order:

1. `rutide-core` to crates.io.
2. `rutide-cli` to crates.io after the core version is visible in the index.
3. Python source distribution and platform wheels to PyPI.
4. The standalone `rutide` executable archives and checksums to the GitHub
   release.

Uploading requires an explicit, protected release action and registry credentials.
Local and pull-request checks only build and inspect packages; they never publish.
The `Build Python release` workflow always builds immutable wheel/source artifacts
for a version tag. Every host-compatible wheel and the source distribution is
installed into a fresh virtual environment and must pass scalar, vector-batch,
reconstruction, and coefficient-persistence smoke tests before upload is
eligible. The cross-built Linux AArch64 wheel is retained as an artifact but
cannot be executed on the x86-64 runner.

Upload occurs only for a manual dispatch on a tag with an explicit
`publish_target` of `testpypi` or `pypi`. Each target has a separate protected
environment and trusted-publisher identity. Stage the first release through
TestPyPI, install that exact version from TestPyPI in a clean environment, then
rerun the same tagged workflow with the PyPI target after approval. The default
`none` target only builds and verifies artifacts.

Python wheels target Linux x86-64/AArch64, macOS x86-64/Apple Silicon, and Windows
x86-64 with the CPython 3.9 stable ABI. Adding another target requires a tested
wheel job; it must not be inferred from a source distribution alone.

## Verify

Install each artifact into a clean environment, run `rutide --version`, import
`rutide`, execute the packaged smoke tests, and confirm the tag, changelog, Cargo
metadata, Python metadata, embedded `__version__`, and coefficient schema all
agree. The Python release workflow automates this for installable distributions;
the CLI archives and cross-built Linux AArch64 wheel still require matching-host
verification before their first public release.
