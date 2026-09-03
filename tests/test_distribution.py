from __future__ import annotations

from email.parser import BytesParser
from email.policy import default
from pathlib import Path
import re
import subprocess
import sys
from zipfile import ZipFile

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_REQUIREMENTS = {
    "httpx",
    "lancedb",
    "lxml",
    "openai",
    "platformdirs",
    "pyarrow",
    "pydantic",
    "pydantic-ai-slim",
    "tantivy",
}


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_dir = tmp_path_factory.mktemp("wheel")
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(output_dir)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(output_dir.glob("vigyan-*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def test_wheel_excludes_checkout_agent_and_legacy_module(built_wheel: Path) -> None:
    with ZipFile(built_wheel) as archive:
        members = set(archive.namelist())

    assert "vigyan/agent/research_agent.py" not in members
    assert not any(member.startswith("vigyan_dev/") for member in members)


def test_wheel_has_exact_normalized_runtime_requirement_names(
    built_wheel: Path,
) -> None:
    with ZipFile(built_wheel) as archive:
        metadata_paths = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        assert len(metadata_paths) == 1
        metadata = BytesParser(policy=default).parsebytes(
            archive.read(metadata_paths[0])
        )

    requirements = metadata.get_all("Requires-Dist", [])
    requirement_names = {
        re.sub(r"[-_.]+", "-", re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", value)[0]).lower()
        for value in requirements
    }
    assert requirement_names == EXPECTED_REQUIREMENTS


def test_wheel_imports_before_the_checkout_from_an_external_directory(
    built_wheel: Path,
    tmp_path: Path,
) -> None:
    script = f"""
import sys
sys.path.insert(0, {str(built_wheel)!r})
import vigyan.agent
assert vigyan.agent.__all__ == ["ResearchCapability", "ResearchRetriever"]
assert {str(built_wheel)!r} in vigyan.agent.__file__
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
