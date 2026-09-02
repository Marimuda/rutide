"""Install one release artifact in a clean environment and run its smoke test."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def environment_python(directory: Path) -> Path:
    """Return the interpreter created by :mod:`venv` on this platform."""
    name = "python.exe" if sys.platform == "win32" else "python"
    return directory / ("Scripts" if sys.platform == "win32" else "bin") / name


def main() -> None:
    if len(sys.argv) != 4 or sys.argv[2] not in {"wheel", "sdist"}:
        raise SystemExit(
            "usage: install_artifact_and_smoke.py DIST_DIRECTORY {wheel|sdist} EXPECTED_VERSION"
        )
    distribution = Path(sys.argv[1]).resolve(strict=True)
    kind = sys.argv[2]
    expected_version = sys.argv[3]
    pattern = "*.whl" if kind == "wheel" else "*.tar.gz"
    artifacts = sorted(distribution.glob(pattern))
    if len(artifacts) != 1:
        raise RuntimeError(
            f"expected exactly one {kind} matching {pattern!r} in {distribution}, "
            f"found {[path.name for path in artifacts]!r}"
        )
    smoke = Path(__file__).with_name("installed_smoke.py").resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="rutide-installed-smoke-") as temporary:
        environment = Path(temporary) / "environment"
        venv.EnvBuilder(with_pip=True, clear=True).create(environment)
        python = environment_python(environment)
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--force-reinstall",
                str(artifacts[0]),
            ],
            check=True,
            cwd=temporary,
        )
        subprocess.run(
            [str(python), str(smoke), expected_version],
            check=True,
            cwd=temporary,
        )


if __name__ == "__main__":
    main()
