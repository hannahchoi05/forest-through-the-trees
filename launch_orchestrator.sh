#!/bin/bash
# Standalone launcher — run with: bash launch_orchestrator.sh

REPO_ROOT="/Users/hannahchoi/forest-through-the-trees"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
ORCHESTRATOR="$REPO_ROOT/Plots/Code/orchestrate_36.py"
ORCH_LOG="$REPO_ROOT/orchestrator.log"
ORCH_PID="$REPO_ROOT/orchestrator.pid"

rm -f "$ORCH_LOG" "$ORCH_PID"

nohup caffeinate -disu \
  "$PYTHON_BIN" "$ORCHESTRATOR" --repo-root "$REPO_ROOT" \
  >"$ORCH_LOG" 2>&1 &

echo $! >"$ORCH_PID"
PID=$(cat "$ORCH_PID")

echo "Background orchestrator PID: $PID"
echo "Main log:  $ORCH_LOG"
echo "PID file:  $ORCH_PID"
echo ""
echo "Monitor with: tail -f '$ORCH_LOG'"
echo "Stop with:    kill $PID"
