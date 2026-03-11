# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Takeshi Nishikawa
#=============================================================================
  FullerMD.jl — Shared module for Fullerene NPT-MD simulations
  Constants, utility functions, I/O, crystal builders, neighbor lists.
  No JACC dependency — pure Julia.
=============================================================================#
module FullerMD

using Printf

# ═══════════ Physical constants ═══════════
# Unit system: A (length), amu (mass), eV (energy), fs (time), K (temperature), GPa (pressure)
const CONV       = 9.64853321e-3    # eV·fs²/(amu·A²) — Newton's law conversion
const kB         = 8.617333262e-5   # Boltzmann constant [eV/K]
const eV2GPa     = 160.21766208     # eV/A³ → GPa
const eV2kcalmol = 23.06054783      # eV → kcal/mol
const mC         = 12.011           # Carbon atomic mass [amu]

# ═══════════ LJ potential parameters (C-C) ═══════════
const sigma_LJ = 3.431              # C-C LJ sigma [A]
const eps_LJ   = 2.635e-3           # C-C LJ epsilon [eV]
const RCUT     = 3.0 * sigma_LJ     # Cutoff ~10.29 A
const RCUT2    = RCUT * RCUT
const sig2_LJ  = sigma_LJ * sigma_LJ
# Shift V(RCUT) for continuity: sigma/RCUT = 1/3
const VSHFT    = 4.0 * eps_LJ * ((1.0/729.0)^2 - 1.0/729.0)

# ═══════════ Array limits ═══════════
const MAX_NATOM = 84
const MAX_NEIGH = 80
const VECTOR_LENGTH = 128

# ═══════════ Structs ═══════════
mutable struct NPTState
    xi::Float64                  # Thermostat friction
    Q::Float64                   # Thermostat mass
    Vg::Vector{Float64}          # Barostat velocity (9-element flat 3x3)
    W::Float64                   # Barostat mass
    Pe::Float64                  # Target pressure [GPa]
    Tt::Float64                  # Target temperature [K]
    Nf::Int                      # Degrees of freedom
end

struct MolData
    coords::Vector{Float64}      # natom*3 body-frame coordinates
    natom::Int
    Rmol::Float64                # Max distance from CoM
    Dmol::Float64                # Max inter-atom distance
    I0::Float64                  # Isotropic moment of inertia
    Mmol::Float64                # Molecular mass [amu]
    bonds::Vector{Tuple{Int,Int}} # Bond pairs (for MMMD)
end

# ═══════════ 3x3 matrix operations (flat 9-element) ═══════════
# Layout: h[1..9] = [H00,H01,H02, H10,H11,H12, H20,H21,H22]
# h[3i+j+1] for 0-based (i,j)

@inline function mat_det9(h)
    @inbounds begin
        h[1]*(h[5]*h[9]-h[6]*h[8]) -
        h[2]*(h[4]*h[9]-h[6]*h[7]) +
        h[3]*(h[4]*h[8]-h[5]*h[7])
    end
end

@inline mat_tr9(h) = @inbounds h[1] + h[5] + h[9]

function mat_inv9!(hi, h)
    d = mat_det9(h)
    id = 1.0 / d
    @inbounds begin
        hi[1] = id*(h[5]*h[9] - h[6]*h[8])
        hi[2] = id*(h[3]*h[8] - h[2]*h[9])
        hi[3] = id*(h[2]*h[6] - h[3]*h[5])
        hi[4] = id*(h[6]*h[7] - h[4]*h[9])
        hi[5] = id*(h[1]*h[9] - h[3]*h[7])
        hi[6] = id*(h[3]*h[4] - h[1]*h[6])
        hi[7] = id*(h[4]*h[8] - h[5]*h[7])
        hi[8] = id*(h[2]*h[7] - h[1]*h[8])
        hi[9] = id*(h[1]*h[5] - h[2]*h[4])
    end
    return nothing
end

# ═══════════ Minimum image convention ═══════════
@inline function mimg(dx, dy, dz, hi, h)
    @inbounds begin
        s0 = hi[1]*dx + hi[2]*dy + hi[3]*dz
        s1 = hi[4]*dx + hi[5]*dy + hi[6]*dz
        s2 = hi[7]*dx + hi[8]*dy + hi[9]*dz
        s0 -= round(s0); s1 -= round(s1); s2 -= round(s2)
        ndx = h[1]*s0 + h[2]*s1 + h[3]*s2
        ndy = h[4]*s0 + h[5]*s1 + h[6]*s2
        ndz = h[7]*s0 + h[8]*s1 + h[9]*s2
    end
    return ndx, ndy, ndz
end

# ═══════════ PBC: wrap positions into cell ═══════════
function apply_pbc!(pos, h, hi, N)
    for i in 1:N
        @inbounds begin
            px = pos[3i-2]; py = pos[3i-1]; pz = pos[3i]
            s0 = hi[1]*px + hi[2]*py + hi[3]*pz
            s1 = hi[4]*px + hi[5]*py + hi[6]*pz
            s2 = hi[7]*px + hi[8]*py + hi[9]*pz
            s0 -= floor(s0); s1 -= floor(s1); s2 -= floor(s2)
            pos[3i-2] = h[1]*s0 + h[2]*s1 + h[3]*s2
            pos[3i-1] = h[4]*s0 + h[5]*s1 + h[6]*s2
            pos[3i]   = h[7]*s0 + h[8]*s1 + h[9]*s2
        end
    end
end

# ═══════════ Quaternion operations ═══════════
# q = [w, x, y, z] stored as qv[4(i-1)+1 .. 4(i-1)+4]

@inline function q2R_vals(w, x, y, z)
    return (1-2(y*y+z*z), 2(x*y-w*z), 2(x*z+w*y),
            2(x*y+w*z), 1-2(x*x+z*z), 2(y*z-w*x),
            2(x*z-w*y), 2(y*z+w*x), 1-2(x*x+y*y))
end

function qmul(aw,ax,ay,az, bw,bx,by,bz)
    return (aw*bw-ax*bx-ay*by-az*bz,
            aw*bx+ax*bw+ay*bz-az*by,
            aw*by-ax*bz+ay*bw+az*bx,
            aw*bz+ax*by-ay*bx+az*bw)
end

function omega2dq(wx, wy, wz, dt)
    wm = sqrt(wx*wx + wy*wy + wz*wz)
    th = wm * dt * 0.5
    if th < 1e-14
        return (1.0, 0.5*dt*wx, 0.5*dt*wy, 0.5*dt*wz)
    else
        s = sin(th) / wm
        return (cos(th), s*wx, s*wy, s*wz)
    end
end

# ═══════════ C60 coordinate generation ═══════════
function generate_c60()
    phi = (1.0 + sqrt(5.0)) / 2.0
    tmp = zeros(60, 3)
    cyc = [1 2 3; 2 3 1; 3 1 2]  # cyclic permutation indices (1-based)
    n = 0
    # Group 1: (0, ±1, ±3φ) cyclic → 12 vertices
    for p in 1:3, s2 in (-1,1), s3 in (-1,1)
        n += 1
        tmp[n, cyc[p,2]] = Float64(s2)
        tmp[n, cyc[p,3]] = s3 * 3.0 * phi
    end
    # Group 2: (±2, ±(1+2φ), ±φ) cyclic → 24 vertices
    for p in 1:3, s1 in (-1,1), s2 in (-1,1), s3 in (-1,1)
        n += 1
        tmp[n, cyc[p,1]] = s1 * 2.0
        tmp[n, cyc[p,2]] = s2 * (1.0 + 2.0*phi)
        tmp[n, cyc[p,3]] = s3 * phi
    end
    # Group 3: (±1, ±(2+φ), ±2φ) cyclic → 24 vertices
    for p in 1:3, s1 in (-1,1), s2 in (-1,1), s3 in (-1,1)
        n += 1
        tmp[n, cyc[p,1]] = Float64(s1)
        tmp[n, cyc[p,2]] = s2 * (2.0 + phi)
        tmp[n, cyc[p,3]] = s3 * 2.0 * phi
    end
    # Center of mass and scale
    cm = [sum(tmp[:,a]) / 60.0 for a in 1:3]
    for i in 1:60, a in 1:3
        tmp[i,a] = (tmp[i,a] - cm[a]) * 0.72
    end
    # Compute radius and moment of inertia
    Rmol = 0.0; Isum = 0.0
    for i in 1:60
        r2 = tmp[i,1]^2 + tmp[i,2]^2 + tmp[i,3]^2
        r = sqrt(r2)
        if r > Rmol; Rmol = r; end
        Isum += mC * r2
    end
    I0 = Isum * 2.0 / 3.0
    Mmol = 60.0 * mC
    # Flatten to 1D
    coords = zeros(60*3)
    for i in 1:60, a in 1:3
        coords[(i-1)*3 + a] = tmp[i, a]
    end
    return MolData(coords, 60, Rmol, 0.0, I0, Mmol, Tuple{Int,Int}[])
end

# ═══════════ Load .cc1 file ═══════════
function load_cc1(path::String)
    lines = readlines(path)
    natom = parse(Int, strip(lines[1]))
    if natom > MAX_NATOM
        error("natom=$natom > MAX_NATOM=$MAX_NATOM")
    end
    tmp = zeros(natom, 3)
    bonds = Tuple{Int,Int}[]
    for i in 1:natom
        parts = split(strip(lines[i+1]))
        tmp[i,1] = parse(Float64, parts[3])
        tmp[i,2] = parse(Float64, parts[4])
        tmp[i,3] = parse(Float64, parts[5])
        # Parse bonds if present (flag + bond indices)
        if length(parts) >= 6
            flag = parse(Int, parts[6])
            for b in 1:flag
                if 6+b <= length(parts)
                    j = parse(Int, parts[6+b])
                    if j > i  # avoid duplicates
                        push!(bonds, (i, j))
                    end
                end
            end
        end
    end
    # Center of mass
    cm = [sum(tmp[:,a]) / natom for a in 1:3]
    for i in 1:natom, a in 1:3
        tmp[i,a] -= cm[a]
    end
    # Radius and diameter
    Rmol = 0.0; Dmol = 0.0
    for i in 1:natom
        r = sqrt(tmp[i,1]^2 + tmp[i,2]^2 + tmp[i,3]^2)
        if r > Rmol; Rmol = r; end
        for j in (i+1):natom
            d = sqrt((tmp[i,1]-tmp[j,1])^2 + (tmp[i,2]-tmp[j,2])^2 + (tmp[i,3]-tmp[j,3])^2)
            if d > Dmol; Dmol = d; end
        end
    end
    # Moment of inertia (isotropic average)
    Ixx = 0.0; Iyy = 0.0; Izz = 0.0
    for i in 1:natom
        r2 = tmp[i,1]^2 + tmp[i,2]^2 + tmp[i,3]^2
        Ixx += mC * (r2 - tmp[i,1]^2)
        Iyy += mC * (r2 - tmp[i,2]^2)
        Izz += mC * (r2 - tmp[i,3]^2)
    end
    I0 = (Ixx + Iyy + Izz) / 3.0
    Mmol = natom * mC
    # Flatten
    coords = zeros(natom * 3)
    for i in 1:natom, a in 1:3
        coords[(i-1)*3 + a] = tmp[i, a]
    end
    return MolData(coords, natom, Rmol, Dmol, I0, Mmol, bonds)
end

# ═══════════ Fullerene resolver ═══════════
function resolve_fullerene(spec::String, lib::String="FullereneLib")
    sl = lowercase(spec)
    if sl in ("buckyball", "c60", "c60:ih")
        return (lib * "/C60-76/C60-Ih.cc1", "C60(Ih)")
    elseif sl in ("c70", "c70:d5h")
        return (lib * "/C60-76/C70-D5h.cc1", "C70(D5h)")
    elseif sl in ("c72", "c72:d6d")
        return (lib * "/C60-76/C72-D6d.cc1", "C72(D6d)")
    elseif sl in ("c74", "c74:d3h")
        return (lib * "/C60-76/C74-D3h.cc1", "C74(D3h)")
    elseif length(sl) > 4 && sl[1:4] == "c76:"
        sym = spec[5:end]
        return (lib * "/C60-76/C76-" * sym * ".cc1", "C76(" * sym * ")")
    elseif length(sl) > 4 && sl[1:4] == "c84:"
        rest = spec[5:end]
        cp = findfirst(':', rest)
        if cp !== nothing
            n = parse(Int, rest[1:cp-1])
            sym = rest[cp+1:end]
            fn = @sprintf("C84-No.%02d-%s.cc1", n, sym)
            return (lib * "/C84/" * fn, "C84 No." * string(n))
        else
            n = parse(Int, rest)
            pfx = @sprintf("C84-No.%02d-", n)
            dpath = lib * "/C84"
            if isdir(dpath)
                for fn in readdir(dpath)
                    if startswith(fn, pfx) && endswith(fn, ".cc1")
                        return (dpath * "/" * fn, "C84 No." * string(n))
                    end
                end
            end
        end
    end
    error("Unknown fullerene: $spec")
end

# ═══════════ Crystal structure builders ═══════════
# All write to flat pos[1:N*3] and h[1:9], return N

function make_fcc!(pos, h, a, nc)
    bas = [0.0 0.0 0.0; 0.5a 0.5a 0.0; 0.5a 0.0 0.5a; 0.0 0.5a 0.5a]
    n = 0
    for ix in 0:nc-1, iy in 0:nc-1, iz in 0:nc-1, b in 1:4
        n += 1
        pos[3n-2] = a*ix + bas[b,1]
        pos[3n-1] = a*iy + bas[b,2]
        pos[3n]   = a*iz + bas[b,3]
    end
    fill!(h, 0.0)
    h[1] = nc*a; h[5] = nc*a; h[9] = nc*a
    return n
end

function make_hcp!(pos, h, a, nc)
    c = a * sqrt(8.0/3.0)
    a1 = [a, 0.0, 0.0]
    a2 = [a/2, a*sqrt(3.0)/2, 0.0]
    a3 = [0.0, 0.0, c]
    bas = [0.0 0.0 0.0; 1.0/3 2.0/3 0.5]
    n = 0
    for ix in 0:nc-1, iy in 0:nc-1, iz in 0:nc-1, b in 1:2
        n += 1
        fx = ix + bas[b,1]; fy = iy + bas[b,2]; fz = iz + bas[b,3]
        pos[3n-2] = fx*a1[1] + fy*a2[1] + fz*a3[1]
        pos[3n-1] = fx*a1[2] + fy*a2[2] + fz*a3[2]
        pos[3n]   = fx*a1[3] + fy*a2[3] + fz*a3[3]
    end
    fill!(h, 0.0)
    h[1] = nc*a1[1]; h[4] = nc*a1[2]
    h[2] = nc*a2[1]; h[5] = nc*a2[2]
    h[9] = nc*a3[3]
    return n
end

function make_bcc!(pos, h, a, nc)
    bas = [0.0 0.0 0.0; 0.5a 0.5a 0.5a]
    n = 0
    for ix in 0:nc-1, iy in 0:nc-1, iz in 0:nc-1, b in 1:2
        n += 1
        pos[3n-2] = a*ix + bas[b,1]
        pos[3n-1] = a*iy + bas[b,2]
        pos[3n]   = a*iz + bas[b,3]
    end
    fill!(h, 0.0)
    h[1] = nc*a; h[5] = nc*a; h[9] = nc*a
    return n
end

function default_a0(dmax, st, scale)
    m = 1.4
    if st == "FCC"
        a0 = dmax * sqrt(2.0) * m
    elseif st == "HCP"
        a0 = dmax * m
    else
        a0 = dmax * 2.0 / sqrt(3.0) * m
    end
    return a0 * scale
end

# ═══════════ Symmetric neighbor list (host) ═══════════
function nlist_build_sym!(nl_count, nl_list, pos, h, hi, N, rmcut, max_neigh)
    rc2 = (rmcut + 3.0)^2
    fill!(nl_count, 0)
    for i in 1:N
        for j in (i+1):N
            dx = pos[3j-2] - pos[3i-2]
            dy = pos[3j-1] - pos[3i-1]
            dz = pos[3j]   - pos[3i]
            dx, dy, dz = mimg(dx, dy, dz, hi, h)
            r2 = dx*dx + dy*dy + dz*dz
            if r2 < rc2
                ci = nl_count[i] + 1
                cj = nl_count[j] + 1
                if ci <= max_neigh
                    nl_list[(i-1)*max_neigh + ci] = j
                    nl_count[i] = ci
                else
                    @printf("WARNING: nl overflow mol %d (count=%d)\n", i, ci)
                end
                if cj <= max_neigh
                    nl_list[(j-1)*max_neigh + cj] = i
                    nl_count[j] = cj
                else
                    @printf("WARNING: nl overflow mol %d (count=%d)\n", j, cj)
                end
            end
        end
    end
end

# ═══════════ NPT state initialization ═══════════
function make_npt(T, Pe, N)
    Nf = 6*N - 3  # translation(3) + rotation(3) per mol - CoM constraint(3)
    xi = 0.0
    Q = max(Nf * kB * T * 100.0^2, 1e-20)
    Vg = zeros(9)
    W = max((Nf + 9) * kB * T * 1000.0^2, 1e-20)
    return NPTState(xi, Q, Vg, W, Pe, T, Nf)
end

# ═══════════ Instantaneous observables ═══════════
@inline inst_T(KE, Nf) = 2.0 * KE / (Nf * kB)
@inline function inst_P(Wm9, KEt, V)
    @inbounds (2.0*KEt + Wm9[1] + Wm9[5] + Wm9[9]) / (3.0*V) * eV2GPa
end

# ═══════════ OVITO XYZ output (rigid body version) ═══════════
function write_ovito_rigid(io, istep, dt, pos, vel, qv, body, h, N, natom)
    @printf(io, "%d\n", N * natom)
    @printf(io, "Lattice=\"%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\" " *
        "Properties=species:S:1:pos:R:3:c_mol:I:1:vx:R:1:vy:R:1:vz:R:1 " *
        "Time=%.4f Step=%d pbc=\"T T T\"\n",
        h[1],h[4],h[7],h[2],h[5],h[8],h[3],h[6],h[9], istep*dt, istep)
    for i in 1:N
        w,x,y,z = qv[4i-3], qv[4i-2], qv[4i-1], qv[4i]
        R11,R12,R13,R21,R22,R23,R31,R32,R33 = q2R_vals(w,x,y,z)
        for a in 1:natom
            bx = body[3a-2]; by = body[3a-1]; bz = body[3a]
            rx = pos[3i-2] + R11*bx + R12*by + R13*bz
            ry = pos[3i-1] + R21*bx + R22*by + R23*bz
            rz = pos[3i]   + R31*bx + R32*by + R33*bz
            @printf(io, "C %14.8f %14.8f %14.8f %5d %14.8e %14.8e %14.8e\n",
                rx, ry, rz, i, vel[3i-2], vel[3i-1], vel[3i])
        end
    end
end

# ═══════════ OVITO XYZ output (all-atom version) ═══════════
function write_ovito_allatom(io, istep, dt, pos, vel, h, Na, mol_id)
    @printf(io, "%d\n", Na)
    @printf(io, "Lattice=\"%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\" " *
        "Properties=species:S:1:pos:R:3:c_mol:I:1:vx:R:1:vy:R:1:vz:R:1 " *
        "Time=%.4f Step=%d pbc=\"T T T\"\n",
        h[1],h[4],h[7],h[2],h[5],h[8],h[3],h[6],h[9], istep*dt, istep)
    for i in 1:Na
        @printf(io, "C %14.8f %14.8f %14.8f %5d %14.8e %14.8e %14.8e\n",
            pos[3i-2], pos[3i-1], pos[3i], mol_id[i],
            vel[3i-2], vel[3i-1], vel[3i])
    end
end

# ═══════════ CLI option parsing ═══════════
function parse_args(args)
    opts = Dict{String,String}()
    for a in args
        if startswith(a, "--")
            eq = findfirst('=', a)
            if eq !== nothing
                k = lowercase(a[3:eq-1])
                v = a[eq+1:end]
            else
                k = lowercase(a[3:end])
                v = ""
            end
            opts[k] = v
        end
    end
    return opts
end

get_opt(opts, key, default) = get(opts, key, default)

# ═══════════ File utilities ═══════════
file_exists(p) = isfile(p)
dir_exists(p) = isdir(p)

function unique_file(base, ext)
    p = base * ext
    if !isfile(p); return p; end
    for n in 1:9999
        c = base * "_" * string(n) * ext
        if !isfile(c); return c; end
    end
    return base * "_9999" * ext
end

function build_suffix(opts)
    defaults = [("fullerene","C60"),("crystal","fcc"),("cell","3"),
                ("temp","298.0"),("pres","0.0"),("step","10000"),
                ("dt","1.0"),("init_scale","1.0"),("seed","42")]
    sfx = ""
    for (key, _) in defaults
        if haskey(opts, key)
            cv = replace(opts[key], ':' => '_')
            if length(cv) >= 2 && cv[end-1:end] == ".0"
                cv = cv[1:end-2]
            end
            sfx *= "_" * key * "_" * cv
        end
    end
    return sfx
end

function restart_filename(oname, istep, nsteps)
    b = oname
    dt = findlast('.', b)
    if dt !== nothing; b = b[1:dt-1]; end
    p = findfirst("ovito_traj", b)
    if p !== nothing
        b = b[1:p.start-1] * "restart" * b[p.stop+1:end]
    end
    if istep == nsteps
        return b * ".rst"
    end
    dg = 1; x = nsteps
    while x >= 10; x = div(x, 10); dg += 1; end
    return b * "_" * lpad(istep, dg, '0') * ".rst"
end

# ═══════════ Exports ═══════════
export CONV, kB, eV2GPa, eV2kcalmol, mC
export sigma_LJ, eps_LJ, RCUT, RCUT2, sig2_LJ, VSHFT
export MAX_NATOM, MAX_NEIGH, VECTOR_LENGTH
export NPTState, MolData
export mat_det9, mat_tr9, mat_inv9!
export mimg, apply_pbc!
export q2R_vals, qmul, omega2dq
export generate_c60, load_cc1, resolve_fullerene
export make_fcc!, make_hcp!, make_bcc!, default_a0
export nlist_build_sym!
export make_npt, inst_T, inst_P
export write_ovito_rigid, write_ovito_allatom
export parse_args, get_opt
export file_exists, dir_exists
export unique_file, build_suffix, restart_filename

end # module FullerMD
