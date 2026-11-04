#!/usr/bin/env bash
# Staged speed-range curriculum: 1 m/s walking policy -> stable tracking in [1, 3] m/s.
#
# Usage (from this directory):
#   bash train_curriculum.sh
#
# Environment knobs:
#   PY=<interpreter>        default: C:/Users/32560/.conda/envs/isaaclab/python.exe
#   NUM_ENVS=1024           parallel environments
#   HEADING_L1=-1.5         optional |heading error| L1 weight (0 disables)
#   YAW_SCALE=1.5           multiply all yaw/heading reward weights (1.0 = default)
#   HEADING_CONTROL=1       enable the action-level PD heading loop for every stage
#   RESET_OBS_NORM=0        re-open the obs-normaliser count on every resume.  Keep 0:
#                           the count must be >= 5e6 or the statistics drift inside one
#                           rollout and corrupt PPO's importance ratios (a 200k reset made
#                           the fall rate go 0 -> 92% in 5 iterations).
#
# How it works:
#   * Stage 1 resumes the 1 m/s policy (vx [0.2, 1.2]) and widens the range only
#     slightly.  Each later stage resumes the newest checkpoint of the previous stage.
#   * Widening by 0.5 m/s avoids jumping straight into a speed region where every
#     rollout falls and PPO has no useful gradient.
#   * --save_interval 50 keeps crash losses small; re-run with
#     `--resume --resume_latest` (or re-run this script) to continue.
#
# NOTE: agent.py's clip_actions must be > 4 for any command above 1.36 m/s -- see the
# comment there.  The staged ranges below assume that fix is in place.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-C:/Users/32560/.conda/envs/isaaclab/python.exe}"
NUM_ENVS="${NUM_ENVS:-1024}"
HEADING_L1="${HEADING_L1:--1.5}"
YAW_SCALE="${YAW_SCALE:-1.0}"
RESET_OBS_NORM="${RESET_OBS_NORM:-0}"

if [[ "${HEADING_CONTROL:-0}" != "0" ]]; then
  export INFANTRY_RL_HEADING_CONTROL=1
fi

NORM_ARGS=()
if [[ "$RESET_OBS_NORM" != "0" ]]; then
  NORM_ARGS=(--reset_obs_norm_count "$RESET_OBS_NORM")
fi

stage () {  # $1=vx_min  $2=vx_max  $3=iterations  $4=run_name
  local vmin="$1" vmax="$2" iters="$3" name="$4"
  local resume=(--resume --resume_latest)
  if [[ "$name" == "vx_1p5" ]]; then
    # boot from the existing 1 m/s checkpoint (it lives in this project's log root)
    resume=(--resume --load_run 2026-10-22_01-17-16 --checkpoint model_1100.pt)
  fi
  echo "=== $name: vx [$vmin, $vmax] m/s, $iters iters, heading_l1=$HEADING_L1 ==="
  "$PY" train.py --task locomotion --headless --num_envs "$NUM_ENVS" \
    "${resume[@]}" "${NORM_ARGS[@]}" \
    --vx_min "$vmin" --vx_max "$vmax" \
    --heading_l1="$HEADING_L1" --yaw_scale="$YAW_SCALE" \
    --iterations "$iters" --save_interval 50 --run_name "$name"
}

stage 0.5 1.5  800  vx_1p5
stage 0.5 2.0  800  vx_2p0
stage 0.5 2.5  800  vx_2p5
stage 1.0 3.0 1500  vx_3p0

echo
echo "Done. Evaluate every point of the range, not just the endpoints:"
echo
echo "  for v in 1.0 1.5 2.0 2.5 3.0; do"
echo "    \"$PY\" play.py --task locomotion --headless --steps 3000 --cmd_vx \$v \\"
echo "      --load_run <newest run> --checkpoint model_<N>.pt"
echo "  done"
