"""GRSTAPS-X deployment knobs: species by robot type, solver timeouts, paths."""

import os
import stat

import pytest

from omniplanner.grstaps_planner import (
    GrstapsDomain,
    _itags_binary,
    _run,
    _solver_params,
    _work_dir,
    species_for,
)


def test_robot_type_species_reads_the_whole_fleet():
    every_spot_has_an_arm = {"spot": "spot_arm", "simulated-spot": "spot_arm"}
    assert species_for("spot", type_overrides=every_spot_has_an_arm) == "spot_arm"
    assert species_for("Simulated-Spot", type_overrides=every_spot_has_an_arm) == (
        "spot_arm"
    )
    # A named robot's override still wins; unknown species are ignored.
    assert (
        species_for("spot", {"euclid": "spot"}, "euclid", every_spot_has_an_arm)
        == "spot"
    )
    assert species_for("spot", type_overrides={"spot": "tank"}) == "spot"


def test_solver_timeouts_are_capped():
    params = _solver_params(GrstapsDomain(solver_timeout_s=30.0))
    assert params["itags_parameters"]["timeout"] == 30.0
    assert params["scheduler_parameters"]["timeout"] == 30.0
    assert params["scheduler_parameters"]["milp_timeout"] == 30.0


def test_a_hung_solver_becomes_a_planning_failure(tmp_path):
    hang = tmp_path / "itags"
    hang.write_text("#!/bin/sh\nsleep 60\n")
    hang.chmod(hang.stat().st_mode | stat.S_IEXEC)
    d = GrstapsDomain(solver_timeout_s=-29.0)  # kill after 1 s
    with pytest.raises(RuntimeError, match="did not finish"):
        _run([str(hang)], d, str(tmp_path))


def test_paths_come_from_config_then_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("ADT4_GRSTAPS_WORK_DIR", str(tmp_path / "env"))
    assert _work_dir(GrstapsDomain()) == str(tmp_path / "env")
    assert _work_dir(GrstapsDomain(work_dir=str(tmp_path / "cfg"))) == str(
        tmp_path / "cfg"
    )
    monkeypatch.delenv("ADT4_ITAGS_BINARY", raising=False)
    fake = tmp_path / "bin" / "itags"
    fake.parent.mkdir()
    fake.write_text("")
    monkeypatch.setenv("PATH", f"{fake.parent}{os.pathsep}{os.environ['PATH']}")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    assert _itags_binary(GrstapsDomain()) == str(fake)
