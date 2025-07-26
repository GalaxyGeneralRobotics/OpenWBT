#!/bin/bash
ISAAC_SIM_PATH="/path-to-isaac-sim-standalone@4.5.0"
PYTHON_EXE="$ISAAC_SIM_PATH/python.sh"

$PYTHON_EXE -m deploy.run_teleoperation_isaacsim_vision --config run_teleoperation.yaml
