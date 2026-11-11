"""Regression tests for where solve_pddl's debug PDDL dumps land.

They used to be written straight into $HOME
(omniplanner_domain_<ts>_<hash>.pddl etc.), littering it on every solve.
resolve_dump_dir() must default to ~/adt4_output/omniplanner/ and honor an
OMNIPLANNER_DUMP_DIR override.

Runnable both under pytest and directly:
    PYTHONPATH=omniplanner/src python omniplanner/tests/test_pddl_dump_dir.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dsg_pddl.pddl_planning import resolve_dump_dir  # noqa: E402


def test_resolve_dump_dir_default(tmp_path, monkeypatch):
    monkeypatch.delenv("OMNIPLANNER_DUMP_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    result = resolve_dump_dir()

    expected = os.path.join(str(tmp_path), "adt4_output", "omniplanner")
    assert result == expected
    assert os.path.isdir(expected)


def test_resolve_dump_dir_env_override(tmp_path, monkeypatch):
    override = tmp_path / "custom_dump_dir"
    monkeypatch.setenv("OMNIPLANNER_DUMP_DIR", str(override))

    result = resolve_dump_dir()

    assert result == str(override)
    assert os.path.isdir(str(override))


def test_resolve_dump_dir_env_override_expands_user(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OMNIPLANNER_DUMP_DIR", "~/custom_via_tilde")

    result = resolve_dump_dir()

    expected = os.path.join(str(tmp_path), "custom_via_tilde")
    assert result == expected
    assert os.path.isdir(expected)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
