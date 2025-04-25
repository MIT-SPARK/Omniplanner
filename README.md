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
