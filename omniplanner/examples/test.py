import numpy as np
from robot_executor_interface.action_descriptions import ActionSequence, Follow

from omniplanner.goto_points import GotoPointsDomain
from omniplanner.omniplanner import (
    PlanRequest,
    full_planning_pipeline,
)


def compile_plan(plan, plan_id, robot_name, frame_id):
    actions = []
    for p in plan.plan:
        xs = np.interp(np.linspace(0, 1, 10), [0, 1], [p.start[0], p.goal[0]])
        ys = np.interp(np.linspace(0, 1, 10), [0, 1], [p.start[1], p.goal[1]])
        p_interp = np.vstack([xs, ys])
        actions.append(Follow(frame=frame_id, path2d=p_interp.T))

    seq = ActionSequence(plan_id=plan_id, robot_name=robot_name, actions=actions)
    return seq


points = np.array(
    [
        [0.03350246, 0.27892633],
        [0.16300951, 0.16012492],
        [0.71635923, 0.5341003],
        [0.8763498, 0.43243519],
        [0.05777218, 0.51004976],
        [0.96980544, 0.00746369],
        [0.53927086, 0.75623442],
        [0.77329046, 0.66824145],
        [0.08683688, 0.49439621],
        [0.87066708, 0.50754294],
    ]
)

# robot_poses = {"spot", self.get_spot_pose}

req = PlanRequest(
    domain=GotoPointsDomain(),
    goal=[1, 2, 3, 4],
    initial_state=np.array([0.0, 0.1]),
)
plan = full_planning_pipeline(req, points)

#        goal = GotoPointsGoal(
#            goal_points=msg.point_names_to_visit, robot_id=msg.robot_id
#        )
#        req = PlanRequest(
#            domain=GotoPointsDomain(),
#            goal=goal,
#            robot_states=robot_poses,
#        )


print("Plan from planning domain:")
print(plan)

compiled_plan = compile_plan(plan, "abc123", "spot", "a_coordinate_frame")
print("compiled plan:")
print(compiled_plan)
