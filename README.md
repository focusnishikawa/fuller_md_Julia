# fuller_md_Julia

Fullerene crystal NPT molecular dynamics simulations in Julia, using [JACC.jl](https://github.com/JuliaORNL/JACC.jl) for portable CPU/GPU execution.

This is a Julia port of the [fuller_md](https://github.com/focusnishikawa/fuller_md) C++ code. All OpenMP/OpenACC compile-time switches are replaced by JACC.jl's runtime backend selection, eliminating code duplication.

## Programs

| Program | Description | Force Field |
|---------|-------------|-------------|
| `fuller_LJ_core.jl` | Core LJ rigid-body MD (fixed params) | LJ intermolecular |
| `fuller_LJ_npt_md.jl` | Full LJ rigid-body NPT-MD | LJ intermolecular |
| `fuller_airebo_npt_md.jl` | AIREBO all-atom NPT-MD | REBO-II + LJ |
| `fuller_LJ_npt_mmmd.jl` | Molecular mechanics NPT-MD | Bond+Angle+Dihedral+Improper+LJ |

## Setup

```bash
# 1. Install JACC and set backend
julia setup.jl threads    # CPU (default)
julia setup.jl cuda       # NVIDIA GPU
julia setup.jl amdgpu     # AMD GPU
julia setup.jl oneapi     # Intel GPU
```

## Usage

```bash
# Using the run script
cd src
./Run_fuller.sh fuller_LJ_npt_md --temp=500 --step=10000

# Or directly with julia
julia --project=.. fuller_LJ_npt_md.jl --temp=500 --step=10000

# Multi-threaded (threads backend)
JULIA_NUM_THREADS=4 julia --project=.. fuller_LJ_npt_md.jl --step=10000
```

### Common Options

All programs accept `--key=value` options:

| Option | Description | Default |
|--------|-------------|---------|
| `--fullerene=<name>` | Fullerene species | C60 |
| `--crystal=<fcc\|hcp\|bcc>` | Crystal structure | fcc |
| `--cell=<N>` | Unit cell repeats | 3 |
| `--temp=<K>` | Target temperature [K] | 298.0 |
| `--pres=<GPa>` | Target pressure [GPa] | 0.0 |
| `--step=<N>` | Production steps | 10000 |
| `--dt=<fs>` | Time step [fs] | 1.0 / 0.5 / 0.1 |
| `--coldstart=<N>` | Cold-start steps at 4K | 0 |
| `--warmup=<N>` | Warm-up ramp steps | 0 |
| `--ovito=<N>` | OVITO XYZ output interval | 0 (off) |
| `--restart=<N>` | Restart save interval | 0 (off) |
| `--resfile=<path>` | Resume from restart | |
| `--libdir=<path>` | FullereneLib directory | FullereneLib |

### Supported Fullerenes

C60 (Ih), C70 (D5h), C72 (D6d), C74 (D3h), C76 (D2, Td), C84 (No.01-24)

### Examples

```bash
# Basic C60 FCC crystal at 298K
./Run_fuller.sh fuller_LJ_npt_md

# High temperature with OVITO output
./Run_fuller.sh fuller_LJ_npt_md --temp=500 --step=50000 --ovito=100

# Cold start + warmup + production
./Run_fuller.sh fuller_airebo_npt_md --coldstart=5000 --warmup=5000 --step=20000

# C84 fullerene with BCC crystal
./Run_fuller.sh fuller_LJ_npt_md --fullerene=C84:5 --crystal=bcc --cell=4

# MM force field with custom parameters
./Run_fuller.sh fuller_LJ_npt_mmmd --ff_kb=500 --ff_kth=70 --step=50000

# Restart save and resume
./Run_fuller.sh fuller_LJ_npt_md --step=50000 --restart=10000
./Run_fuller.sh fuller_LJ_npt_md --resfile=restart_LJ_julia_010000.rst
```

### Stop Control

Create a directory during execution to control the simulation:
- `mkdir abort.md` — Stop immediately (saves restart if enabled)
- `mkdir stop.md` — Stop at next restart checkpoint

## Testing

```bash
cd src
./Test_fuller.sh
```

## Project Structure

```
fuller_md_Julia/
├── Project.toml              # Julia project definition
├── LocalPreferences.toml.example  # JACC backend config example
├── setup.jl                  # One-time setup script
├── README.md
├── .gitignore
├── FullereneLib/             # Fullerene coordinate library
│   ├── C60-76/
│   └── C84/
└── src/
    ├── FullerMD.jl           # Shared module (constants, I/O, utilities)
    ├── fuller_LJ_core.jl     # Core LJ rigid-body MD
    ├── fuller_LJ_npt_md.jl   # Full LJ rigid-body NPT-MD
    ├── fuller_airebo_npt_md.jl  # AIREBO NPT-MD
    ├── fuller_LJ_npt_mmmd.jl # Molecular mechanics NPT-MD
    ├── Run_fuller.sh          # Execution helper
    └── Test_fuller.sh         # Validation tests
```

## Unit System

Angstrom (length), amu (mass), eV (energy), fs (time), K (temperature), GPa (pressure)

## License

This project is licensed under the [BSD 3-Clause License](LICENSE).

## Other Versions

| Repository | Language | Description |
|-----------|----------|-------------|
| [fuller_md](https://github.com/focusnishikawa/fuller_md) | C++ (Japanese) | C++ version, Japanese |
| [fuller_md_en](https://github.com/focusnishikawa/fuller_md_en) | C++ (English) | C++ version, English |
| [fuller_md_Julia](https://github.com/focusnishikawa/fuller_md_Julia) | Julia (English) | This repository |
| [fuller_md_Julia_ja](https://github.com/focusnishikawa/fuller_md_Julia_ja) | Julia (Japanese) | Julia version, Japanese |
| [fuller_md_fortran](https://github.com/focusnishikawa/fuller_md_fortran) | Fortran 95 (Japanese) | Fortran version, Japanese |
| [fuller_md_fortran_en](https://github.com/focusnishikawa/fuller_md_fortran_en) | Fortran 95 (English) | Fortran version, English |
