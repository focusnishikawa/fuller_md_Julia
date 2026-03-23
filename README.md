# fuller_md_Julia — Fullerene Crystal NPT Molecular Dynamics (Julia)

NPT molecular dynamics simulation codes for C60/C70/C72/C74/C76/C84 fullerene crystals (Julia).
Three force field models (LJ rigid-body, Molecular Mechanics, AIREBO) with
[JACC.jl](https://github.com/JuliaORNL/JACC.jl) for portable CPU/GPU execution.
Ported from the [C++ version fuller_md](https://github.com/focusnishikawa/fuller_md).

## Directory Structure

```
fuller_md_Julia/
├── README.md
├── Project.toml              ← Julia project definition
├── LocalPreferences.toml.example  ← JACC backend config example
├── setup.jl                  ← One-time setup script
├── FullereneLib/             ← Fullerene molecular coordinate data (.cc1)
│   ├── C60-76/               ←   C60(Ih), C70(D5h), C72(D6d), C74(D3h), C76(D2,Td)
│   └── C84/                  ←   C84 No.01–No.24 (24 isomers)
└── src/
    ├── FullerMD.jl            ← Shared module (constants, I/O, utilities)
    ├── fuller_LJ_core.jl      [1] LJ rigid-body core
    ├── fuller_LJ_npt_md.jl    [2] LJ rigid-body full NPT-MD
    ├── fuller_LJ_npt_mmmd.jl  [3] Molecular mechanics full NPT-MD
    ├── fuller_airebo_npt_md.jl [4] AIREBO NPT-MD
    ├── Run_fuller.sh           ← Execution helper script
    └── Test_fuller.sh          ← Validation test script
```

## Source Files

### Core Version [1] — Learning & Benchmarking

Parameters fixed in source code (T=300K, P=0GPa, dt=1fs, 1000 steps).
Only `nc` (cell size) as argument.

| # | File | Description |
|---|------|-------------|
| 1 | `fuller_LJ_core.jl` | LJ rigid-body core. CPU/GPU switching via JACC.jl |

### Full Versions [2][3][4] — Production Runs

All runtime options supported. Restart save/resume, OVITO XYZ output, stop control (abort.md/stop.md).

| # | File | Force Field | Default dt |
|---|------|-------------|-----------|
| 2 | `fuller_LJ_npt_md.jl` | LJ rigid-body intermolecular | 1.0 fs |
| 3 | `fuller_LJ_npt_mmmd.jl` | Molecular mechanics (Bond+Angle+Dihedral+Improper+LJ) | 0.1 fs |
| 4 | `fuller_airebo_npt_md.jl` | AIREBO (REBO-II + LJ) | 0.5 fs |

## Setup

```bash
# Install JACC and set backend
julia setup.jl threads    # CPU (default)
julia setup.jl cuda       # NVIDIA GPU
julia setup.jl amdgpu     # AMD GPU
julia setup.jl oneapi     # Intel GPU
```

## Usage

### Core Version

```bash
cd src
./Run_fuller.sh fuller_LJ_core
./Run_fuller.sh fuller_LJ_core --cell=5
```

### Full Versions — Basic Execution

```bash
cd src

# LJ rigid-body (default: C60, FCC 3x3x3, 298K, 0GPa, 10000 steps)
./Run_fuller.sh fuller_LJ_npt_md

# Specify temperature, pressure, and steps
./Run_fuller.sh fuller_LJ_npt_md --temp=500 --pres=1.0 --step=50000

# Cold start (4K) + warmup + production
./Run_fuller.sh fuller_LJ_npt_md --coldstart=2000 --warmup=3000 --step=20000

# Multi-threaded (threads backend)
JULIA_NUM_THREADS=4 julia --project=.. fuller_LJ_npt_md.jl --step=10000
```

### Full Versions — OVITO Output

```bash
./Run_fuller.sh fuller_LJ_npt_md --step=10000 --ovito=100
./Run_fuller.sh fuller_LJ_npt_mmmd --step=20000 --ovito=200
./Run_fuller.sh fuller_airebo_npt_md --step=10000 --ovito=100
```

### Full Versions — Restart

```bash
# Save restart
./Run_fuller.sh fuller_LJ_npt_md --step=50000 --restart=10000

# Resume from restart file
./Run_fuller.sh fuller_LJ_npt_md --resfile=restart_LJ_julia_010000.rst
```

### Full Versions — Stop Control

```bash
mkdir abort.md    # Stop immediately (saves restart if enabled)
mkdir stop.md     # Stop at next restart checkpoint
```

## Validation Tests

```bash
cd src
./Test_fuller.sh
```

## Runtime Options (Full Versions)

| Option | Description | Default |
|--------|-------------|---------|
| `--fullerene=<name>` | Fullerene species | C60 |
| `--crystal=<fcc\|hcp\|bcc>` | Crystal structure | fcc |
| `--cell=<N>` | Unit cell repeats | 3 |
| `--temp=<K>` | Target temperature [K] | 298.0 |
| `--pres=<GPa>` | Target pressure [GPa] | 0.0 |
| `--step=<N>` | Production steps | 10000 |
| `--dt=<fs>` | Time step [fs] | Force-field dependent |
| `--coldstart=<N>` | Cold-start steps at 4K | 0 |
| `--warmup=<N>` | Warmup ramp steps 4K→T | 0 |
| `--ovito=<N>` | OVITO XYZ output interval (0=off) | 0 |
| `--restart=<N>` | Restart save interval (0=off) | 0 |
| `--resfile=<path>` | Resume from restart file | — |
| `--libdir=<path>` | Fullerene library directory | FullereneLib |

### Molecular Mechanics [3] Additional Options

| Option | Description | Default |
|--------|-------------|---------|
| `--ff_kb=<kcal/mol>` | Bond stretching force constant | 469.0 |
| `--ff_kth=<kcal/mol>` | Angle bending force constant | 63.0 |
| `--ff_v2=<kcal/mol>` | Dihedral torsion force constant | 14.5 |
| `--ff_kimp=<kcal/mol>` | Improper dihedral force constant | 15.0 |

## Supported Fullerenes

C60 (Ih), C70 (D5h), C72 (D6d), C74 (D3h), C76 (D2, Td), C84 (No.01-24)

## Physical Model

- **Ensemble**: NPT (isothermal-isobaric)
- **Thermostat**: Nose-Hoover chain
- **Barostat**: Parrinello-Rahman
- **Time integration**: Velocity-Verlet
- **Periodic boundary conditions**: 3D (triclinic cell)
- **Neighbor list**: Symmetric full list

### LJ Rigid-Body [1][2]
- Intermolecular: Lennard-Jones (C-C, sigma=3.4A, epsilon=2.63meV)
- Rigid-body rotation: Quaternion representation

### Molecular Mechanics [3]
- Intramolecular: Bond stretching + Angle bending + Dihedral + Improper
- Intermolecular: LJ + Coulomb
- All-atom DOF

### AIREBO [4]
- REBO-II (Brenner 2002): Covalent bonds (bond-order potential)
- LJ (Stuart 2000): Intermolecular van der Waals
- All-atom DOF

## Unit System

| Quantity | Unit |
|----------|------|
| Length | A (Angstrom) |
| Mass | amu (atomic mass unit) |
| Energy | eV (electron volt) |
| Time | fs (femtosecond) |
| Temperature | K (Kelvin) |
| Pressure | GPa (gigapascal) |

## Supported Environments

- Julia 1.9+ (JACC.jl compatible)
- CPU: threads backend
- NVIDIA GPU: CUDA backend
- AMD GPU: AMDGPU backend
- Intel GPU: oneAPI backend

## License

This project is licensed under the [BSD 3-Clause License](LICENSE).

## Other Languages & Versions

| Repository | Language | Description |
|-----------|----------|-------------|
| [fuller_md](https://github.com/focusnishikawa/fuller_md) | C++ (Japanese) | C++ Japanese |
| [fuller_md_en](https://github.com/focusnishikawa/fuller_md_en) | C++ (English) | C++ English |
| [fuller_md_Julia](https://github.com/focusnishikawa/fuller_md_Julia) | Julia (English) | This repository |
| [fuller_md_Julia_ja](https://github.com/focusnishikawa/fuller_md_Julia_ja) | Julia (Japanese) | Julia Japanese |
| [fuller_md_fortran](https://github.com/focusnishikawa/fuller_md_fortran) | Fortran 95 (Japanese) | Fortran Japanese |
| [fuller_md_fortran_en](https://github.com/focusnishikawa/fuller_md_fortran_en) | Fortran 95 (English) | Fortran English |
