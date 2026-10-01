# Remaining deployment work

The native planner and ROS adapter are connected. The GRSTAPS-specific
Heracles agent, repair launch, and ITAGS overlay already exist; the old claim
that prompt/launch wiring was entirely missing is obsolete.

Before calling the whole live stack verified:

1. **Extend semantics only with matching execution.** The adapter now rejects
   `or`, `at-*`, `holding`, unknown predicates and unsupported negations, and
   the GRSTAPS prompt documents this subset. Add these features only with
   corresponding task/executor models. Invalid constraint symbols are still
   warned and skipped. Inspection emits `Gaze`; it does not itself establish
   a perception-backed `safe` fact.
2. **Validate live fleet repair.** Run the actual agent → database → goal manager
   → planner → simulator/hardware flow. Verify robot names, TF, arm availability,
   coalition size, pause/resume, failures, and replacement plans. The repair
   overlay's `hilbert: spot_arm` is a simulation assumption. A one-robot fleet
   cannot satisfy its two-ground-robot inspection requirement.
3. **Harden lifecycle and progress.** READY/DONE publication is volatile; late
   subscribers/restarts can miss progress. Failed tasks stop locally but do not
   automatically cancel/replan the fleet. The goal manager tracks requested
   constraints before planning succeeds and does not use plan-ID-correlated
   transactional updates. Its horizon compares nominal schedule offsets, not
   elapsed execution time. Pausing one addressed executor is not a fleet barrier.
4. **Isolate solver jobs.** Concurrent requests must use different scenario
   names. Add per-request directories, subprocess cancellation/time limits,
   stronger solution validation, and cleanup. Upstream cache/resume support is
   not a persistent warm-start service in this adapter: each request starts a
   solver subprocess.
5. **Extend physical models as needed.** The adapter shares a 2D ground graph
   across species; it does not provide UAV control, inter-robot collision
   avoidance, hard deadlines, or physical capability detection. Split
   manipulation (`atomic_manipulation=False`) does not ensure one carrier from
   pick through place; use the default atomic mode.

Completed corrections and reproducible tests are recorded in the
[integration audit](integration-audit.md). The [README](../README.md) is the
concise description of supported operation.
