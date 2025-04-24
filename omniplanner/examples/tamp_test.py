import matplotlib.pyplot as plt
import numpy as np
from robot_executor_interface.action_descriptions import ActionSequence, Follow

from omniplanner.tamp_planner import TAMPDomain, TAMPProblem, TAMPPlan, TAMPGoal
from omniplanner.omniplanner import (
    PlanRequest,
    full_planning_pipeline,
)
from omniplanner_ros.omniplanner_node import temp_compile_tamp_plan

import spark_dsg
from dsg_tamp.mapping.outdoor_dsg_utils import spark_dsg_to_tamp
from dsg_tamp.mapping.outdoor_dsg_viz import plot_dsg_objects, plot_dsg_places


goal = TAMPGoal(
    goal="(and (Holding o1))",
    robot_id="spot",
)
req = PlanRequest(
    domain=TAMPDomain(),
    goal=goal,
    robot_states={"spot":np.array([2, -4, 1.0])},
)

traversable_semantics=["ground", "water", "sidewalk", "road", "floor", "surface"]
dsg_filepath = "/home/rrg/dsg-tamp/data/scene_graphs/building45_loop.json"
dsg_filepath = "/home/rrg/data/dcist/hydra/2025-04-22-14_51_18_dsg.json"
dsg_filepath = "/home/rrg/data/dcist/hydra/2025-04-22-19_47_11_dsg.json"

G = spark_dsg.DynamicSceneGraph.load(dsg_filepath)
dsg = spark_dsg_to_tamp(
    G, traversable_semantics=traversable_semantics
)
plan = full_planning_pipeline(req, dsg)
print("Plan from planning domain:")
print(plan)

commands = temp_compile_tamp_plan(dsg, plan, 0, 'spot', 'map')
print(commands)
waypoints = np.empty((0, 3))
for command in commands.actions:
    if type(command) is Follow:
        waypoints = np.vstack((waypoints, command.path2d))

for p in plan:
    if p.name == "pick":
        pr = p.args[1].pose
        plt.scatter([pr[0]], [pr[1]], color="r")
    elif p.name == "place_nearby":
        pr = p.args[1].pose
        plt.scatter([pr[0]], [pr[1]], color="c")
        po = p.args[2].pose
        plt.scatter([po[0]], [po[1]], color="b")


wps = np.array(waypoints)

plot_dsg_places(
    dsg, with_edges=True, plot_mask=dsg.places.traversable, plot_indices=True
)
plot_dsg_objects(dsg)
plt.plot(wps[:, 0], wps[:, 1])

plt.show()
