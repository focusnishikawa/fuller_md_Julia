#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Takeshi Nishikawa
#=============================================================================
# Run_fuller.sh — Execute Julia fullerene MD simulations
#
# Usage:
#   ./Run_fuller.sh <program> [options]
#
# Programs:
#   fuller_LJ_core         Core LJ rigid-body MD (fixed parameters)
#   fuller_LJ_npt_md       Full LJ rigid-body NPT-MD
#   fuller_airebo_npt_md   AIREBO (REBO-II + LJ) NPT-MD
#   fuller_LJ_npt_mmmd     Molecular mechanics + LJ NPT-MD
#
# Examples:
#   ./Run_fuller.sh fuller_LJ_core --cell=3
#   ./Run_fuller.sh fuller_LJ_npt_md --temp=500 --step=10000
#   ./Run_fuller.sh fuller_airebo_npt_md --temp=300 --step=5000
#   ./Run_fuller.sh fuller_LJ_npt_mmmd --temp=300 --dt=0.1 --step=20000
#
# Environment:
#   JULIA_NUM_THREADS=N    Set number of threads for JACC threads backend
#=============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <program> [options]"
    echo "Programs: fuller_LJ_core, fuller_LJ_npt_md, fuller_airebo_npt_md, fuller_LJ_npt_mmmd"
    exit 1
fi

PROG="$1"
shift

exec julia --project="$BASE_DIR" "$SCRIPT_DIR/${PROG}.jl" "$@"
