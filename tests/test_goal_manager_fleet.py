"""Willow's goal manager: per-robot plan state, achieved facts, reloads, and
repair versus additional tasks. The base station is willow; the fleet has three
Spots and a Husky (gauss)."""

import pytest

pytest.importorskip("robot_executor_msgs.msg")

from goal_manager_harness import Harness  # noqa: E402
from std_msgs.msg import String  # noqa: E402

MISSION = "(and (visited-object o1) (visited-object o2) (visited-object o3))"


@pytest.fixture
def h(tmp_path):
    harness = Harness(tmp_path)
    yield harness
    harness.close()


def _three_robot_mission(h):
    h.send(MISSION)
    h.plan(
        euclid=("pe", 2, ["p1", "o1"]),
        hamilton=("ph", 2, ["p2", "o2"]),
        hilbert=("pi", 2, ["p3", "o3"]),
    )
    h.goals.clear()
    h.sent.clear()


def test_a_finished_robot_makes_its_targets_done_while_others_stay_in_progress(h):
    _three_robot_mission(h)
    h.done("euclid", "pe")
    h.done("euclid", "pe")  # euclid finished; hamilton and hilbert still running
    assert "euclid" not in h.node._robots
    assert set(h.node._robots) == {"hamilton", "hilbert"}

    h.send("(visited-object o1)")
    assert h.goals == [] and h.sent == []  # done: no plan, nobody paused
    assert "already done" in h.texts[-1] and "o1 by euclid" in h.texts[-1]

    # o2 is hamilton's and still in progress: not replanned, not called done.
    h.send("(and (visited-object o1) (visited-object o2))")
    assert h.goals == [] and h.sent == []
    assert "in progress (hamilton)" in h.texts[-1]
    assert "already done" not in h.texts[-1]
    assert any("(visited-object o2) = in progress (hamilton)" in m for m in h.logs)


def test_a_finished_plan_does_not_cover_a_target_it_never_reached(h):
    """P16: once a plan is over, only what was achieved counts, not what the
    plan would have covered."""
    h.send("(visited-object o1)")
    h.plan(euclid=("pe", 2, ["p1", "o1"]))
    h.done("euclid", "pe", success=False, action="FOLLOW")  # never got to o1
    h.goals.clear()
    h.send("(visited-object o1)")
    assert len(h.goals) == 1
    assert h.goals[0].goal.pddl_goal == "(visited-object o1)"


def test_a_failed_robot_gives_back_only_its_conjuncts(h):
    _three_robot_mission(h)
    h.done("euclid", "pe")
    h.done("euclid", "pe")
    h.done("hamilton", "ph", success=False, action="PICK", obj="o2")
    assert "hamilton: PICK o2 failed" in h.texts[-1]
    assert "(visited-object o2)" in h.texts[-1]
    assert h.node._lost == {"(visited-object o2)"}

    h.send(MISSION)
    (sent,) = h.goals
    # o1 is done and left out; o2 lost its robot; o3 is still hilbert's.
    assert sent.goal.pddl_goal == "(and (visited-object o2) (visited-object o3))"
    assert h.busy(sent) == []  # a repair: the planner may interrupt anyone
    log = [m for m in h.logs if m.startswith("replanning (repair)")][-1]
    assert "lost its robot" in log
    assert "(visited-object o2) = new" in log
    assert "(visited-object o3) = in progress (hilbert)" in log


def test_a_replan_request_sends_only_what_is_left(h):
    _three_robot_mission(h)
    h.done("euclid", "pe")
    h.done("euclid", "pe")
    h.node._replan_request_cb(String(data="scene changed: o3 moved"))
    (sent,) = h.goals
    assert sent.goal.pddl_goal == "(and (visited-object o2) (visited-object o3))"
    assert h.busy(sent) == []


def test_a_reload_clears_everything(h):
    h.map_version(5)
    _three_robot_mission(h)
    h.done("euclid", "pe")
    h.done("euclid", "pe")
    h.map_version(1)  # reload: the version went down
    assert h.node._robots == {} and h.node._achieved == {}
    assert "map was reloaded" in h.texts[-1]
    h.map_version(5)  # the old number again, on the new map
    h.send("(visited-object o1)")
    assert len(h.goals) == 1  # not done any more: planned on the new map


def test_no_resume_of_an_old_plan_after_a_reload(tmp_path):
    h = Harness(tmp_path, resume_on_plan_failure=True)
    h.map_version(3)
    h.send("(visited-object o1)")
    h.plan(euclid=("pe", 2, ["o1"]))
    h.map_version(2)  # reload
    h.send("(visited-object o4)")
    h.sent.clear()
    h.failed("no plan")
    assert ("euclid", "stop") in h.sent and ("euclid", "resume") not in h.sent
    assert "map was reloaded since the current plan" in h.texts[-1]
    h.close()


def test_a_goal_revising_the_mission_is_a_repair_without_a_busy_list(h):
    _three_robot_mission(h)
    h.send(MISSION, forbidden=["o7"])  # same goal, new constraints
    (sent,) = h.goals
    assert h.busy(sent) == []
    assert {("euclid", "pause"), ("hamilton", "pause")} <= set(h.sent)
    assert any(
        "the planner may replace the plans of euclid" in m for m in h.logs
    )  # P14: who is mid-task


def test_a_goal_overlapping_the_mission_is_a_repair(h):
    _three_robot_mission(h)
    h.send("(and (visited-object o2) (visited-object o9))")
    (sent,) = h.goals
    assert h.busy(sent) == []
    assert any("revises the running mission" in m for m in h.logs)


def test_an_additional_task_lists_the_busy_robots(h):
    _three_robot_mission(h)
    h.send("(visited-object o9)")
    (sent,) = h.goals
    assert h.busy(sent) == [["euclid", "hamilton", "hilbert"]]
    assert h.sent == []  # busy robots carry on: nobody is paused
    # Its plan, for the Husky alone, does not stop the busy robots.
    h.plan(gauss=("", 0, ["o9"]))
    assert h.sent == [] and set(h.node._robots) >= {"euclid", "hamilton", "hilbert"}
    assert "(visited-object o9)" in h.node._mission


def test_an_additional_task_naming_a_busy_robot_leaves_it_off_the_list(h):
    _three_robot_mission(h)
    h.send("(visited-object-by hamilton o9)")
    (sent,) = h.goals
    assert h.busy(sent) == [["euclid", "hilbert"]]


def test_a_stalled_robot_is_reported_once(h):
    """P11: a plan accepted, and no action result within progress_timeout_s."""
    h.send("(visited-object o1)")
    h.plan(euclid=("c9019ff2aa", 6, ["o1"]))
    h.t += 30
    h.node._check_progress()
    assert not any("no progress" in t for t in h.texts)
    h.t += 31
    h.node._check_progress()
    h.t += 60
    h.node._check_progress()
    warned = [t for t in h.texts if "no progress" in t]
    assert len(warned) == 1 and warned[0].startswith("euclid:")
    assert "c9019ff2" in warned[0]
    assert h.node._lost == {"(visited-object o1)"}
    # The same goal again goes to the planner: its robot was lost.
    h.goals.clear()
    h.send("(visited-object o1)")
    assert len(h.goals) == 1


def test_a_paused_robot_is_not_stalled(h):
    h.send("(visited-object o1)")
    h.plan(euclid=("pe", 6, ["o1"]))
    h.node._pause_executor()
    h.t += 120
    h.node._check_progress()
    assert not any("no progress" in t for t in h.texts)


def test_a_goal_the_planner_finds_satisfied_is_reported_and_not_replanned(h):
    """P8."""
    h.send("(object-in-place o58 t2260)")
    h.failed("goal already satisfied by the current state")
    assert "already holds; nothing to do" in h.texts[-1]
    h.goals.clear()
    h.send("(object-in-place o58 t2260)")
    assert h.goals == []  # no replanning loop
    assert "already done" in h.texts[-1]
    h.map_version(1)  # an edit: placements are judged again
    h.send("(object-in-place o58 t2260)")
    assert len(h.goals) == 1


def test_all_empty_plans_mean_the_goal_already_holds(h):
    h.send("(object-in-place o58 t2260)")
    h.plan(euclid=("pe", 0, []), hamilton=("ph", 0, []))
    assert "already holds: the plan has nothing to do" in h.texts[-1]


def test_a_fleet_goal_is_never_called_a_robot(h):
    h.send("(visited-object o1)")
    h.node._plan_robots_cb(String(data='["euclid", "willow"]'))
    assert "willow" not in h.node._plan_robots
    h.node._pause_executor()
    assert ("willow", "pause") not in h.sent
    h.failed("no plan")
    assert "the fleet goal '(visited-object o1)'" in h.texts[-1]
    assert "willow" not in h.texts[-1]


def test_unreached_robots_are_not_claimed_stopped(h):
    _three_robot_mission(h)
    h.unreachable = {"hilbert"}
    h.send("(visited-object o2)", forbidden=["o7"])  # repair: all paused
    h.failed("no plan")
    text = h.texts[-1]
    assert "hilbert" not in text.split(";")[0]
    assert "Stopped euclid and hamilton" in text
    assert "could not reach hilbert" in text


def test_a_husky_cannot_pause_and_is_stopped_by_an_empty_path(tmp_path):
    """P4, with the real executor commands, on robots no live stack has."""
    import rclpy
    from nav_msgs.msg import Path

    husky, spot = "gm_test_husky", "gm_test_spot"
    h = Harness(tmp_path, record_executor=False, spots=[spot], huskies=[husky])
    n = h.node
    got = []
    sub = n.create_subscription(
        Path, f"/{husky}/omniplanner_node/compiled_plan_out", got.append, 10
    )
    h.send("(visited-object o1)")
    h.plan(**{husky: ("", 0, ["o1"]), spot: ("pe", 2, ["p1"])})
    for _ in range(50):  # let the Path publisher match the listener
        if n._executor_pub(husky, "stop").get_subscription_count():
            break
        rclpy.spin_once(n, timeout_sec=0.05)
    n._pause_executor()
    assert any(f"Pause is not supported on a Husky: {husky}" in m for m in h.logs)
    stopped, unreached = n._stop_executor()
    for _ in range(50):
        if got:
            break
        rclpy.spin_once(n, timeout_sec=0.05)
    assert stopped == [husky] and unreached == [spot]  # no Spot executor here
    assert got and list(got[-1].poses) == []
    n.destroy_subscription(sub)
    h.close()


def _path_status(h, robot, status):
    h.node._path_status_cb(
        robot, String(data=f'{{"plan_id": "(path)", "status": "{status}"}}')
    )


def test_a_husky_is_busy_until_the_planner_says_its_path_is_done(h):
    """Found in the local e2e: a Husky reports no actions, so it stayed
    "running" after reaching the end of its Path -- never done, never free."""
    h.send("(and (visited-object o1) (visited-object o3))")
    h.plan(euclid=("pe", 1, ["o1"]), gauss=("", 0, ["o3"]))
    h.goals.clear()
    # While it drives, it is busy: an additional task leaves it alone.
    h.send("(visited-object o9)")
    (sent,) = h.goals
    assert h.busy(sent) == [["euclid", "gauss"]]
    h.goals.clear()
    _path_status(h, "gauss", "progress")
    _path_status(h, "gauss", "done")
    assert "gauss" not in h.node._robots
    assert "o3" in h.node._visited()
    h.send("(visited-object o3)")
    assert h.goals == [] and any("already done" in t for t in h.texts)


def test_a_husky_that_stops_moving_gives_its_part_back(h):
    h.send("(visited-object o3)")
    h.plan(gauss=("", 0, ["o3"]))
    _path_status(h, "gauss", "stalled")
    assert h.node._lost == {"(visited-object o3)"}
    assert any(t.startswith("gauss:") for t in h.texts)


def test_a_husky_sent_an_empty_path_is_free(h):
    """Found in the local e2e: the planner stops an idle Husky with an empty
    Path, and the goal manager counted it busy, so an additional task that
    only it could do was refused ("every robot ... is busy")."""
    from nav_msgs.msg import Path

    h.send("(visited-object o1)")
    h.node._path_cb("gauss", Path())  # the Path comes first ...
    h.plan(euclid=("pe", 2, ["o1"]), gauss=("", 0, []))
    assert "gauss" not in h.node._robots
    h.goals.clear()
    h.send("(visited-object o9)")
    (sent,) = h.goals
    assert h.busy(sent) == [["euclid"]]
    # ... or after the plan that lists it.
    h.t += 60
    h.plan(gauss=("", 0, ["o9"]))
    assert "gauss" in h.node._robots
    h.node._path_cb("gauss", Path())
    assert "gauss" not in h.node._robots
