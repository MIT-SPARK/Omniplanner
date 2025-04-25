# Omniplanner

Omniplanner provides an interface to solving DSG-grounded planning problems
with a variety of solvers and grounding mechanisms. The goal is to enable
module design of command grounding and planning implementations, and a clean
hook for transforming the output of a planner into robot-compatible input.

This repo is still under construction, and details are subject to change.

## Architecture

The Omniplanner architecture can be thought of in two halves: The Omniplanner
ROS node that provides an interface for combining planning commands, scene
representations, and robot commands, and the Omniplanner non-ROS code that
defines the generic interfaces that a planner or language grounding system
needs to implement to work with the Omniplanner node.

The planner plugins that Omniplanner loads are defined in a yaml file like
[this one](https://github.com/MIT-SPARK/Awesome-DCIST-T4/blob/feature/generic_omniplanner/dcist_launch_system/config/bag/omniplanner_plugins.yaml). The omniplanner node
takes the path to this [as a rosparam](https://github.com/MIT-SPARK/Awesome-DCIST-T4/blob/feature/generic_omniplanner/dcist_launch_system/config/bag/omniplanner_node.yaml).

Implementing a planner plugin requires implementing Omniplanner's
`ground_problem` and `make_plan` interface. At planning time, various planning
and grounding methods can be combined through dispatching on the input and
output types of these modules. You need to implement [an interface like
this](https://github.com/MIT-SPARK/Omniplanner/blob/feature/full_genericization/omniplanner/src/omniplanner/goto_points.py)
to give the omniplanner node a hook into your planning and grounding behavior.

The second thing you need to do is implement the ROS hook. This entails
writing [a class like this](https://github.com/MIT-SPARK/Omniplanner/blob/feature/full_genericization/omniplanner_ros/src/omniplanner_ros/goto_points_ros.py)
with a `get_plan_callback` function. You also need to implement the config
registration at the bottom of the file to enable constructing the plugin
based on the plugin YAML definition.

Currently, there is a final step of importing your custom config into the
Omniplanner Node
[here](https://github.com/MIT-SPARK/Omniplanner/blob/de84ccf5d5f71b6f41b04d9bceb24a11eaeb1fe5/omniplanner_ros/src/omniplanner_ros/omniplanner_node.py#L28),
but the intention is to do automatic plugin discovery. Automatic plugin
discovery will enable all downstream planning plugins to be implemented without
touching the omniplanner node.
