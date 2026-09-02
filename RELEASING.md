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
4. Update `compatibility/feature-matrix-v1.json` and verify any NetCDF/JSON schema
   changes have incremented their independent schema constants.
5. Run the full Rust, Python, documentation, package, and wheel checks documented
   in the repository README.
6. Commit the release, create the signed `vX.Y.Z` tag, and push both commit and
   tag. Tags must point at a clean commit whose reported package versions agree.

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
for a version tag. PyPI upload occurs only for a manual dispatch on a tag with the
`publish` input enabled, after approval by the `pypi` environment. Configure that
environment and its PyPI trusted-publisher identity before the first release.

Python wheels target Linux x86-64/AArch64, macOS x86-64/Apple Silicon, and Windows
x86-64 with the CPython 3.9 stable ABI. Adding another target requires a tested
wheel job; it must not be inferred from a source distribution alone.

## Verify

Install each artifact into a clean environment, run `rutide --version`, import
`rutide`, execute the packaged smoke tests, and confirm the tag, changelog, Cargo
metadata, Python metadata, and embedded `__version__` all agree.
