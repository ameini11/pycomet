#!/bin/bash -i

# Wait for system to stabilize after reboot
# sleep 300

# Change directory
file="$(realpath "$0")"
path="$(dirname "$file")"
cd $path

# Launch each Python script in background and redirect logs
./ode/main.py > log.ode 2>&1 &
PID1=$!
./sur/main.py > log.sur 2>&1 &
PID2=$!

echo $PID1 $PID2 > log.pids
echo "All jobs started in background."
