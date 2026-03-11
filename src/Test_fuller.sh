#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Takeshi Nishikawa
#=============================================================================
# Test_fuller.sh — Validation tests for Julia fullerene MD simulations
#
# Runs all 4 simulation programs with minimal steps and checks:
#   - Exit code 0
#   - OVITO output generated (when enabled)
#   - Restart save/load cycle
#
# Usage:
#   ./Test_fuller.sh [--quick]
#     --quick: Run with fewer steps (default: already minimal)
#=============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
JULIA="julia --project=$BASE_DIR"

PASS=0; FAIL=0; TOTAL=0
TMPDIR=$(mktemp -d)

pass() { echo "  [PASS] $1"; PASS=$((PASS+1)); TOTAL=$((TOTAL+1)); }
fail() { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); TOTAL=$((TOTAL+1)); }
run_test() {
    local name="$1"; shift
    echo "--- $name ---"
    if eval "$@" > "$TMPDIR/out_${name}.log" 2>&1; then
        pass "$name"
    else
        fail "$name (exit code $?)"
        tail -5 "$TMPDIR/out_${name}.log"
    fi
}

cd "$TMPDIR"

echo "========================================================================"
echo "  Fuller MD Julia — Validation Tests"
echo "  Working directory: $TMPDIR"
echo "========================================================================"
echo ""

# --- 1. Core LJ ---
echo "=== 1. fuller_LJ_core ==="
run_test "core_basic" "$JULIA $SCRIPT_DIR/fuller_LJ_core.jl --cell=2"

# --- 2. Full LJ ---
echo ""
echo "=== 2. fuller_LJ_npt_md ==="
run_test "lj_basic" "$JULIA $SCRIPT_DIR/fuller_LJ_npt_md.jl --cell=2 --step=100 --libdir=$BASE_DIR/FullereneLib"

# LJ with OVITO output
run_test "lj_ovito" "$JULIA $SCRIPT_DIR/fuller_LJ_npt_md.jl --cell=2 --step=100 --ovito=50 --libdir=$BASE_DIR/FullereneLib"
if ls ovito_traj_LJ_*.xyz 1>/dev/null 2>&1; then
    pass "lj_ovito_file_exists"
else
    fail "lj_ovito_file_exists (no xyz file)"
fi

# LJ restart cycle
run_test "lj_restart_save" "$JULIA $SCRIPT_DIR/fuller_LJ_npt_md.jl --cell=2 --step=100 --restart=50 --libdir=$BASE_DIR/FullereneLib"
RST=$(ls restart_LJ_*.rst 2>/dev/null | head -1)
if [ -n "$RST" ]; then
    pass "lj_restart_file_exists"
    run_test "lj_restart_load" "$JULIA $SCRIPT_DIR/fuller_LJ_npt_md.jl --cell=2 --step=200 --resfile=$RST --libdir=$BASE_DIR/FullereneLib"
else
    fail "lj_restart_file_exists (no rst file)"
fi

# --- 3. AIREBO ---
echo ""
echo "=== 3. fuller_airebo_npt_md ==="
run_test "airebo_basic" "$JULIA $SCRIPT_DIR/fuller_airebo_npt_md.jl --cell=2 --step=50 --libdir=$BASE_DIR/FullereneLib"

run_test "airebo_ovito" "$JULIA $SCRIPT_DIR/fuller_airebo_npt_md.jl --cell=2 --step=50 --ovito=25 --libdir=$BASE_DIR/FullereneLib"
if ls ovito_traj_airebo_*.xyz 1>/dev/null 2>&1; then
    pass "airebo_ovito_file_exists"
else
    fail "airebo_ovito_file_exists (no xyz file)"
fi

# --- 4. MMMD ---
echo ""
echo "=== 4. fuller_LJ_npt_mmmd ==="
run_test "mmmd_basic" "$JULIA $SCRIPT_DIR/fuller_LJ_npt_mmmd.jl --cell=2 --step=50 --libdir=$BASE_DIR/FullereneLib"

run_test "mmmd_ovito" "$JULIA $SCRIPT_DIR/fuller_LJ_npt_mmmd.jl --cell=2 --step=50 --ovito=25 --libdir=$BASE_DIR/FullereneLib"
if ls ovito_traj_mmmd_*.xyz 1>/dev/null 2>&1; then
    pass "mmmd_ovito_file_exists"
else
    fail "mmmd_ovito_file_exists (no xyz file)"
fi

# --- Summary ---
echo ""
echo "========================================================================"
echo "  Results: $PASS passed, $FAIL failed, $TOTAL total"
echo "========================================================================"

# Cleanup
rm -rf "$TMPDIR"

[ $FAIL -eq 0 ] && exit 0 || exit 1
