# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Takeshi Nishikawa
#=============================================================================
  fuller_LJ_npt_mmmd.jl — Fullerene Crystal NPT-MD
  (Molecular Mechanics Force Field + LJ Intermolecular)
  Portable CPU/GPU version using JACC.jl

  V_total = V_bond + V_angle + V_dihedral + V_improper + V_LJ + V_Coulomb

  Usage:
    julia --project=.. fuller_LJ_npt_mmmd.jl [options]

  Options:
    --help                  Show help
    --fullerene=<name>      Fullerene species (default: C60)
    --crystal=<fcc|hcp|bcc> Crystal structure (default: fcc)
    --cell=<nc>             Unit cell repeats (default: 3)
    --temp=<K>              Target temperature [K] (default: 298.0)
    --pres=<GPa>            Target pressure [GPa] (default: 0.0)
    --step=<N>              Production steps (default: 10000)
    --dt=<fs>               Time step [fs] (default: 0.1)
    --init_scale=<s>        Lattice scale factor (default: 1.0)
    --seed=<n>              Random seed (default: 42)
    --coldstart=<N>         Cold-start steps at 4K (default: 0)
    --warmup=<N>            Warm-up ramp steps 4K->T (default: 0)
    --from=<step>           Averaging start step (default: auto)
    --to=<step>             Averaging end step (default: nsteps)
    --mon=<N>               Monitor print interval (default: auto)
    --warmup_mon=<mode>     Warmup monitor: norm|freq|some (default: norm)
    --ovito=<N>             OVITO XYZ output interval, 0=off (default: 0)
    --ofile=<filename>      OVITO output filename (default: auto)
    --restart=<N>           Restart save interval, 0=off (default: 0)
    --resfile=<path>        Resume from restart file
    --libdir=<path>         Fullerene library dir (default: FullereneLib)
    --ff_kb=<kcal/mol>      Bond stretch constant (default: 469.0)
    --ff_kth=<kcal/mol>     Angle bend constant (default: 63.0)
    --ff_v2=<kcal/mol>      Dihedral constant (default: 14.5)
    --ff_kimp=<kcal/mol>    Improper constant (default: 15.0)

  Stop control: create abort.md/ or stop.md/ directory during execution.
  Unit system: A, amu, eV, fs, K, GPa
=============================================================================#

using JACC
using Printf, Random
JACC.@init_backend

include("FullerMD.jl")
using .FullerMD

const kcal2eV = 1.0 / eV2kcalmol
const COULOMB_K = 14.3996
const MAX_LJ_NEIGH_MM = 400

# ═══════════ Topology builder (host) ═══════════

function compute_phi0(cc, i, j, k, l)
    b1x=cc[3j-2]-cc[3i-2]; b1y=cc[3j-1]-cc[3i-1]; b1z=cc[3j]-cc[3i]
    b2x=cc[3k-2]-cc[3j-2]; b2y=cc[3k-1]-cc[3j-1]; b2z=cc[3k]-cc[3j]
    b3x=cc[3l-2]-cc[3k-2]; b3y=cc[3l-1]-cc[3k-1]; b3z=cc[3l]-cc[3k]
    mx=b1y*b2z-b1z*b2y; my=b1z*b2x-b1x*b2z; mz=b1x*b2y-b1y*b2x
    nx=b2y*b3z-b2z*b3y; ny=b2z*b3x-b2x*b3z; nz=b2x*b3y-b2y*b3x
    mm=mx*mx+my*my+mz*mz; nn=nx*nx+ny*ny+nz*nz
    (mm < 1e-20 || nn < 1e-20) && return 0.0
    b2len=sqrt(b2x*b2x+b2y*b2y+b2z*b2z)
    cosphi=clamp((mx*nx+my*ny+mz*nz)/sqrt(mm*nn), -1.0, 1.0)
    sinphi=(mx*b3x+my*b3y+mz*b3z)*b2len/sqrt(mm*nn)
    return atan(sinphi, cosphi)
end

function build_flat_topology(mol, Nmol, kb, kth, v2_dih, k_imp)
    na = mol.natom
    # Build adjacency from bonds
    adj = [Int[] for _ in 1:na]
    for (bi, bj) in mol.bonds
        push!(adj[bi], bj); push!(adj[bj], bi)
    end

    # Per-molecule bonds
    bnd_pairs = Tuple{Int,Int}[]; bnd_r0_v = Float64[]
    for (bi, bj) in mol.bonds
        dx=mol.coords[3bi-2]-mol.coords[3bj-2]; dy=mol.coords[3bi-1]-mol.coords[3bj-1]; dz=mol.coords[3bi]-mol.coords[3bj]
        push!(bnd_pairs, (bi, bj)); push!(bnd_r0_v, sqrt(dx*dx+dy*dy+dz*dz))
    end
    nb_mol = length(bnd_pairs)

    # Per-molecule angles: for each central atom j with neighbors i,k
    ang_triples = Tuple{Int,Int,Int}[]; ang_th0_v = Float64[]
    for j in 1:na
        nj = adj[j]
        for ii in 1:length(nj), kk in ii+1:length(nj)
            i = nj[ii]; k = nj[kk]
            rji = [mol.coords[3i-2]-mol.coords[3j-2], mol.coords[3i-1]-mol.coords[3j-1], mol.coords[3i]-mol.coords[3j]]
            rjk = [mol.coords[3k-2]-mol.coords[3j-2], mol.coords[3k-1]-mol.coords[3j-1], mol.coords[3k]-mol.coords[3j]]
            dji = sqrt(sum(rji.^2)); djk = sqrt(sum(rjk.^2))
            costh = clamp(sum(rji .* rjk) / (dji*djk), -1.0, 1.0)
            push!(ang_triples, (i, j, k)); push!(ang_th0_v, acos(costh))
        end
    end
    nang_mol = length(ang_triples)

    # Per-molecule dihedrals: for each bond j-k, find i-j...k-l
    dih_quads = Tuple{Int,Int,Int,Int}[]; dih_gamma_v = Float64[]
    for (bj, bk) in mol.bonds
        for i in adj[bj]
            i == bk && continue
            for l in adj[bk]
                (l == bj || l == i) && continue
                phi0 = compute_phi0(mol.coords, i, bj, bk, l)
                push!(dih_quads, (i, bj, bk, l)); push!(dih_gamma_v, 2*phi0+π)
            end
        end
    end
    ndih_mol = length(dih_quads)

    # Per-molecule impropers: atoms with exactly 3 neighbors
    imp_quads = Tuple{Int,Int,Int,Int}[]; imp_gamma_v = Float64[]
    for i in 1:na
        if length(adj[i]) == 3
            j, k, l = adj[i][1], adj[i][2], adj[i][3]
            psi0 = compute_phi0(mol.coords, j, i, k, l)
            push!(imp_quads, (i, j, k, l)); push!(imp_gamma_v, 2*psi0+π)
        end
    end
    nimp_mol = length(imp_quads)

    # Total counts
    Nb = nb_mol * Nmol; Nang = nang_mol * Nmol; Ndih = ndih_mol * Nmol; Nimp = nimp_mol * Nmol

    # Allocate flat arrays
    b_i0=zeros(Int32,Nb); b_i1=zeros(Int32,Nb); b_kb_a=zeros(Float64,Nb); b_r0_a=zeros(Float64,Nb)
    ang_i0=zeros(Int32,Nang); ang_i1=zeros(Int32,Nang); ang_i2=zeros(Int32,Nang)
    ang_kth_a=zeros(Float64,Nang); ang_th0_a=zeros(Float64,Nang)
    dih_i0=zeros(Int32,Ndih); dih_i1=zeros(Int32,Ndih); dih_i2=zeros(Int32,Ndih); dih_i3=zeros(Int32,Ndih)
    dih_Vn=zeros(Float64,Ndih); dih_mult=zeros(Int32,Ndih); dih_gamma_a=zeros(Float64,Ndih)
    imp_i0=zeros(Int32,Nimp); imp_i1=zeros(Int32,Nimp); imp_i2=zeros(Int32,Nimp); imp_i3=zeros(Int32,Nimp)
    imp_ki_a=zeros(Float64,Nimp); imp_gamma_a=zeros(Float64,Nimp)
    mol_id_a=zeros(Int32, Nmol*na)

    # Replicate across molecules
    for m in 1:Nmol
        off = (m-1)*na; ob = (m-1)*nb_mol; oa = (m-1)*nang_mol; od = (m-1)*ndih_mol; oi = (m-1)*nimp_mol
        for a in 1:na; mol_id_a[off+a] = m; end
        for b in 1:nb_mol
            b_i0[ob+b] = bnd_pairs[b][1]+off; b_i1[ob+b] = bnd_pairs[b][2]+off
            b_kb_a[ob+b] = kb; b_r0_a[ob+b] = bnd_r0_v[b]
        end
        for a in 1:nang_mol
            ang_i0[oa+a]=ang_triples[a][1]+off; ang_i1[oa+a]=ang_triples[a][2]+off; ang_i2[oa+a]=ang_triples[a][3]+off
            ang_kth_a[oa+a]=kth; ang_th0_a[oa+a]=ang_th0_v[a]
        end
        for d in 1:ndih_mol
            dih_i0[od+d]=dih_quads[d][1]+off; dih_i1[od+d]=dih_quads[d][2]+off
            dih_i2[od+d]=dih_quads[d][3]+off; dih_i3[od+d]=dih_quads[d][4]+off
            dih_Vn[od+d]=v2_dih; dih_mult[od+d]=2; dih_gamma_a[od+d]=dih_gamma_v[d]
        end
        for p in 1:nimp_mol
            imp_i0[oi+p]=imp_quads[p][1]+off; imp_i1[oi+p]=imp_quads[p][2]+off
            imp_i2[oi+p]=imp_quads[p][3]+off; imp_i3[oi+p]=imp_quads[p][4]+off
            imp_ki_a[oi+p]=k_imp; imp_gamma_a[oi+p]=imp_gamma_v[p]
        end
    end
    @printf("  Topology/mol: %d bonds, %d angles, %d dihedrals, %d impropers\n", nb_mol, nang_mol, ndih_mol, nimp_mol)
    @printf("  Total:        %d bonds, %d angles, %d dihedrals, %d impropers\n", Nb, Nang, Ndih, Nimp)
    return (Nb=Nb, b_i0=b_i0, b_i1=b_i1, b_kb=b_kb_a, b_r0=b_r0_a,
            Nang=Nang, ang_i0=ang_i0, ang_i1=ang_i1, ang_i2=ang_i2, ang_kth=ang_kth_a, ang_th0=ang_th0_a,
            Ndih=Ndih, dih_i0=dih_i0, dih_i1=dih_i1, dih_i2=dih_i2, dih_i3=dih_i3, dih_Vn=dih_Vn, dih_mult=dih_mult, dih_gamma=dih_gamma_a,
            Nimp=Nimp, imp_i0=imp_i0, imp_i1=imp_i1, imp_i2=imp_i2, imp_i3=imp_i3, imp_ki=imp_ki_a, imp_gamma=imp_gamma_a,
            mol_id=mol_id_a)
end

# ═══════════ LJ neighbor list (host, half-list, intermolecular) ═══════════
function build_nlist_lj_mm!(nlc, nll, pos, h, hi, Na, mol_id)
    rc2 = (RCUT + 2.0)^2
    fill!(nlc, 0)
    for i in 1:Na-1
        mi = mol_id[i]
        for j in i+1:Na
            mol_id[j] == mi && continue
            dx=pos[3j-2]-pos[3i-2]; dy=pos[3j-1]-pos[3i-1]; dz=pos[3j]-pos[3i]
            dx, dy, dz = mimg(dx, dy, dz, hi, h)
            dx*dx+dy*dy+dz*dz >= rc2 && continue
            ci = nlc[i] + 1
            if ci <= MAX_LJ_NEIGH_MM; nll[(i-1)*MAX_LJ_NEIGH_MM+ci] = j; nlc[i] = ci; end
        end
    end
end

# ═══════════ JACC Kernels ═══════════

function zero_kernel!(i, F)
    @inbounds begin; F[3i-2]=0.0; F[3i-1]=0.0; F[3i]=0.0; end; return nothing
end

function pbc_kernel!(i, pos, h, hi)
    @inbounds begin
        px=pos[3i-2];py=pos[3i-1];pz=pos[3i]
        s0=hi[1]*px+hi[2]*py+hi[3]*pz; s1=hi[4]*px+hi[5]*py+hi[6]*pz; s2=hi[7]*px+hi[8]*py+hi[9]*pz
        s0-=floor(s0);s1-=floor(s1);s2-=floor(s2)
        pos[3i-2]=h[1]*s0+h[2]*s1+h[3]*s2; pos[3i-1]=h[4]*s0+h[5]*s1+h[6]*s2; pos[3i]=h[7]*s0+h[8]*s1+h[9]*s2
    end; return nothing
end

# 1. Bond stretching kernel
function bond_kernel!(b, F, Epbuf, Wbuf, pos, h, hi, b_i0, b_i1, b_kb, b_r0)
    @inbounds begin
        i = b_i0[b]; j = b_i1[b]
        dx=pos[3j-2]-pos[3i-2]; dy=pos[3j-1]-pos[3i-1]; dz=pos[3j]-pos[3i]
        s0=hi[1]*dx+hi[2]*dy+hi[3]*dz; s1=hi[4]*dx+hi[5]*dy+hi[6]*dz; s2=hi[7]*dx+hi[8]*dy+hi[9]*dz
        s0-=round(s0);s1-=round(s1);s2-=round(s2)
        dx=h[1]*s0+h[2]*s1+h[3]*s2; dy=h[4]*s0+h[5]*s1+h[6]*s2; dz=h[7]*s0+h[8]*s1+h[9]*s2
        r = sqrt(dx*dx+dy*dy+dz*dz)
        if r > 1e-10
            dr = r - b_r0[b]
            Epbuf[b] = 0.5 * b_kb[b] * dr * dr
            fm = -b_kb[b] * dr / r
            fx=fm*dx; fy=fm*dy; fz=fm*dz
            @atomic F[3i-2] += fx; @atomic F[3i-1] += fy; @atomic F[3i] += fz
            @atomic F[3j-2] -= fx; @atomic F[3j-1] -= fy; @atomic F[3j] -= fz
            Wbuf[3b-2] = dx*fx; Wbuf[3b-1] = dy*fy; Wbuf[3b] = dz*fz
        else
            Epbuf[b] = 0.0; Wbuf[3b-2]=0.0; Wbuf[3b-1]=0.0; Wbuf[3b]=0.0
        end
    end; return nothing
end

# 2. Angle bending kernel
function angle_kernel!(a, F, Epbuf, pos, h, hi, ang_i0, ang_i1, ang_i2, ang_kth, ang_th0)
    @inbounds begin
        i=ang_i0[a]; j=ang_i1[a]; k=ang_i2[a]
        rji0=pos[3i-2]-pos[3j-2]; rji1=pos[3i-1]-pos[3j-1]; rji2=pos[3i]-pos[3j]
        rjk0=pos[3k-2]-pos[3j-2]; rjk1=pos[3k-1]-pos[3j-1]; rjk2=pos[3k]-pos[3j]
        # mimg rji
        s0=hi[1]*rji0+hi[2]*rji1+hi[3]*rji2; s1=hi[4]*rji0+hi[5]*rji1+hi[6]*rji2; s2=hi[7]*rji0+hi[8]*rji1+hi[9]*rji2
        s0-=round(s0);s1-=round(s1);s2-=round(s2)
        rji0=h[1]*s0+h[2]*s1+h[3]*s2; rji1=h[4]*s0+h[5]*s1+h[6]*s2; rji2=h[7]*s0+h[8]*s1+h[9]*s2
        # mimg rjk
        s0=hi[1]*rjk0+hi[2]*rjk1+hi[3]*rjk2; s1=hi[4]*rjk0+hi[5]*rjk1+hi[6]*rjk2; s2=hi[7]*rjk0+hi[8]*rjk1+hi[9]*rjk2
        s0-=round(s0);s1-=round(s1);s2-=round(s2)
        rjk0=h[1]*s0+h[2]*s1+h[3]*s2; rjk1=h[4]*s0+h[5]*s1+h[6]*s2; rjk2=h[7]*s0+h[8]*s1+h[9]*s2

        dji=sqrt(rji0*rji0+rji1*rji1+rji2*rji2); djk=sqrt(rjk0*rjk0+rjk1*rjk1+rjk2*rjk2)
        if dji < 1e-10 || djk < 1e-10; Epbuf[a]=0.0; return nothing; end
        costh = clamp((rji0*rjk0+rji1*rjk1+rji2*rjk2)/(dji*djk), -0.999999, 0.999999)
        th=acos(costh); dth=th-ang_th0[a]
        Epbuf[a] = 0.5*ang_kth[a]*dth*dth
        sinth=sqrt(1.0-costh*costh+1e-30)
        dV=-ang_kth[a]*dth/sinth

        fi0=dV*(rjk0/(dji*djk)-costh*rji0/(dji*dji)); fk0=dV*(rji0/(dji*djk)-costh*rjk0/(djk*djk))
        fi1=dV*(rjk1/(dji*djk)-costh*rji1/(dji*dji)); fk1=dV*(rji1/(dji*djk)-costh*rjk1/(djk*djk))
        fi2=dV*(rjk2/(dji*djk)-costh*rji2/(dji*dji)); fk2=dV*(rji2/(dji*djk)-costh*rjk2/(djk*djk))

        @atomic F[3i-2]+=fi0; @atomic F[3i-1]+=fi1; @atomic F[3i]+=fi2
        @atomic F[3k-2]+=fk0; @atomic F[3k-1]+=fk1; @atomic F[3k]+=fk2
        @atomic F[3j-2]-=fi0+fk0; @atomic F[3j-1]-=fi1+fk1; @atomic F[3j]-=fi2+fk2
    end; return nothing
end

# 3. Dihedral torsion kernel (Bekker analytical forces)
function dihedral_kernel!(d, F, Epbuf, pos, h, hi, dih_i0, dih_i1, dih_i2, dih_i3, dih_Vn, dih_mult, dih_gamma)
    @inbounds begin
        i0=dih_i0[d]; i1=dih_i1[d]; i2=dih_i2[d]; i3=dih_i3[d]
        b1x=pos[3i1-2]-pos[3i0-2]; b1y=pos[3i1-1]-pos[3i0-1]; b1z=pos[3i1]-pos[3i0]
        b2x=pos[3i2-2]-pos[3i1-2]; b2y=pos[3i2-1]-pos[3i1-1]; b2z=pos[3i2]-pos[3i1]
        b3x=pos[3i3-2]-pos[3i2-2]; b3y=pos[3i3-1]-pos[3i2-1]; b3z=pos[3i3]-pos[3i2]
        # mimg b1,b2,b3
        for_each_mimg = ((b1x,b1y,b1z), (b2x,b2y,b2z), (b3x,b3y,b3z))
        s0=hi[1]*b1x+hi[2]*b1y+hi[3]*b1z;s1=hi[4]*b1x+hi[5]*b1y+hi[6]*b1z;s2=hi[7]*b1x+hi[8]*b1y+hi[9]*b1z
        s0-=round(s0);s1-=round(s1);s2-=round(s2);b1x=h[1]*s0+h[2]*s1+h[3]*s2;b1y=h[4]*s0+h[5]*s1+h[6]*s2;b1z=h[7]*s0+h[8]*s1+h[9]*s2
        s0=hi[1]*b2x+hi[2]*b2y+hi[3]*b2z;s1=hi[4]*b2x+hi[5]*b2y+hi[6]*b2z;s2=hi[7]*b2x+hi[8]*b2y+hi[9]*b2z
        s0-=round(s0);s1-=round(s1);s2-=round(s2);b2x=h[1]*s0+h[2]*s1+h[3]*s2;b2y=h[4]*s0+h[5]*s1+h[6]*s2;b2z=h[7]*s0+h[8]*s1+h[9]*s2
        s0=hi[1]*b3x+hi[2]*b3y+hi[3]*b3z;s1=hi[4]*b3x+hi[5]*b3y+hi[6]*b3z;s2=hi[7]*b3x+hi[8]*b3y+hi[9]*b3z
        s0-=round(s0);s1-=round(s1);s2-=round(s2);b3x=h[1]*s0+h[2]*s1+h[3]*s2;b3y=h[4]*s0+h[5]*s1+h[6]*s2;b3z=h[7]*s0+h[8]*s1+h[9]*s2

        mx=b1y*b2z-b1z*b2y; my=b1z*b2x-b1x*b2z; mz=b1x*b2y-b1y*b2x
        nx=b2y*b3z-b2z*b3y; ny=b2z*b3x-b2x*b3z; nz=b2x*b3y-b2y*b3x
        mm=mx*mx+my*my+mz*mz; nn=nx*nx+ny*ny+nz*nz
        if mm < 1e-20 || nn < 1e-20; Epbuf[d]=0.0; return nothing; end
        imm=1.0/mm; inn=1.0/nn
        b2len=sqrt(b2x*b2x+b2y*b2y+b2z*b2z)
        cosphi=clamp((mx*nx+my*ny+mz*nz)/sqrt(mm*nn), -1.0, 1.0)
        sinphi=(mx*b3x+my*b3y+mz*b3z)*b2len/sqrt(mm*nn)
        phi=atan(sinphi, cosphi)
        mult=dih_mult[d]; gamma=dih_gamma[d]; Vn=dih_Vn[d]
        Epbuf[d] = 0.5*Vn*(1.0+cos(mult*phi-gamma))
        dphi = -0.5*Vn*mult*sin(mult*phi-gamma)

        f1x=dphi*b2len*imm*mx; f1y=dphi*b2len*imm*my; f1z=dphi*b2len*imm*mz
        f4x=-dphi*b2len*inn*nx; f4y=-dphi*b2len*inn*ny; f4z=-dphi*b2len*inn*nz
        b1db2=b1x*b2x+b1y*b2y+b1z*b2z; b2db3=b2x*b3x+b2y*b3y+b2z*b3z
        coef2i=b1db2/(b2len*b2len); coef2k=b2db3/(b2len*b2len)
        f2x=-f1x+coef2i*f1x-coef2k*f4x; f2y=-f1y+coef2i*f1y-coef2k*f4y; f2z=-f1z+coef2i*f1z-coef2k*f4z
        f3x=-f4x-coef2i*f1x+coef2k*f4x; f3y=-f4y-coef2i*f1y+coef2k*f4y; f3z=-f4z-coef2i*f1z+coef2k*f4z

        @atomic F[3i0-2]+=f1x; @atomic F[3i0-1]+=f1y; @atomic F[3i0]+=f1z
        @atomic F[3i1-2]+=f2x; @atomic F[3i1-1]+=f2y; @atomic F[3i1]+=f2z
        @atomic F[3i2-2]+=f3x; @atomic F[3i2-1]+=f3y; @atomic F[3i2]+=f3z
        @atomic F[3i3-2]+=f4x; @atomic F[3i3-1]+=f4y; @atomic F[3i3]+=f4z
    end; return nothing
end

# 4. Improper dihedral kernel (center=i0, neighbors=i1,i2,i3; dihedral order i1-i0-i2-i3)
function improper_kernel!(p, F, Epbuf, pos, h, hi, imp_i0, imp_i1, imp_i2, imp_i3, imp_ki, imp_gamma)
    @inbounds begin
        c0=imp_i0[p]; c1=imp_i1[p]; c2=imp_i2[p]; c3=imp_i3[p]
        b1x=pos[3c0-2]-pos[3c1-2]; b1y=pos[3c0-1]-pos[3c1-1]; b1z=pos[3c0]-pos[3c1]
        b2x=pos[3c2-2]-pos[3c0-2]; b2y=pos[3c2-1]-pos[3c0-1]; b2z=pos[3c2]-pos[3c0]
        b3x=pos[3c3-2]-pos[3c2-2]; b3y=pos[3c3-1]-pos[3c2-1]; b3z=pos[3c3]-pos[3c2]
        # mimg
        s0=hi[1]*b1x+hi[2]*b1y+hi[3]*b1z;s1=hi[4]*b1x+hi[5]*b1y+hi[6]*b1z;s2=hi[7]*b1x+hi[8]*b1y+hi[9]*b1z
        s0-=round(s0);s1-=round(s1);s2-=round(s2);b1x=h[1]*s0+h[2]*s1+h[3]*s2;b1y=h[4]*s0+h[5]*s1+h[6]*s2;b1z=h[7]*s0+h[8]*s1+h[9]*s2
        s0=hi[1]*b2x+hi[2]*b2y+hi[3]*b2z;s1=hi[4]*b2x+hi[5]*b2y+hi[6]*b2z;s2=hi[7]*b2x+hi[8]*b2y+hi[9]*b2z
        s0-=round(s0);s1-=round(s1);s2-=round(s2);b2x=h[1]*s0+h[2]*s1+h[3]*s2;b2y=h[4]*s0+h[5]*s1+h[6]*s2;b2z=h[7]*s0+h[8]*s1+h[9]*s2
        s0=hi[1]*b3x+hi[2]*b3y+hi[3]*b3z;s1=hi[4]*b3x+hi[5]*b3y+hi[6]*b3z;s2=hi[7]*b3x+hi[8]*b3y+hi[9]*b3z
        s0-=round(s0);s1-=round(s1);s2-=round(s2);b3x=h[1]*s0+h[2]*s1+h[3]*s2;b3y=h[4]*s0+h[5]*s1+h[6]*s2;b3z=h[7]*s0+h[8]*s1+h[9]*s2

        mx=b1y*b2z-b1z*b2y; my=b1z*b2x-b1x*b2z; mz=b1x*b2y-b1y*b2x
        nx=b2y*b3z-b2z*b3y; ny=b2z*b3x-b2x*b3z; nz=b2x*b3y-b2y*b3x
        mm=mx*mx+my*my+mz*mz; nn=nx*nx+ny*ny+nz*nz
        if mm < 1e-20 || nn < 1e-20; Epbuf[p]=0.0; return nothing; end
        imm=1.0/mm; inn=1.0/nn
        b2len=sqrt(b2x*b2x+b2y*b2y+b2z*b2z)
        cosphi=clamp((mx*nx+my*ny+mz*nz)/sqrt(mm*nn), -1.0, 1.0)
        sinphi=(mx*b3x+my*b3y+mz*b3z)*b2len/sqrt(mm*nn)
        phi=atan(sinphi, cosphi)
        ki=imp_ki[p]; gm=imp_gamma[p]
        Epbuf[p] = 0.5*ki*(1.0+cos(2*phi-gm))
        dphi = -0.5*ki*2*sin(2*phi-gm)

        f1x=dphi*b2len*imm*mx; f1y=dphi*b2len*imm*my; f1z=dphi*b2len*imm*mz
        f4x=-dphi*b2len*inn*nx; f4y=-dphi*b2len*inn*ny; f4z=-dphi*b2len*inn*nz
        b1db2=b1x*b2x+b1y*b2y+b1z*b2z; b2db3=b2x*b3x+b2y*b3y+b2z*b3z
        c2i=b1db2/(b2len*b2len); c2k=b2db3/(b2len*b2len)
        # Atom mapping: atom1->c1, atom2->c0, atom3->c2, atom4->c3
        @atomic F[3c1-2]+=f1x; @atomic F[3c1-1]+=f1y; @atomic F[3c1]+=f1z
        f2x=-f1x+c2i*f1x-c2k*f4x; f2y=-f1y+c2i*f1y-c2k*f4y; f2z=-f1z+c2i*f1z-c2k*f4z
        @atomic F[3c0-2]+=f2x; @atomic F[3c0-1]+=f2y; @atomic F[3c0]+=f2z
        f3x=-f4x-c2i*f1x+c2k*f4x; f3y=-f4y-c2i*f1y+c2k*f4y; f3z=-f4z-c2i*f1z+c2k*f4z
        @atomic F[3c2-2]+=f3x; @atomic F[3c2-1]+=f3y; @atomic F[3c2]+=f3z
        @atomic F[3c3-2]+=f4x; @atomic F[3c3-1]+=f4y; @atomic F[3c3]+=f4z
    end; return nothing
end

# 5. LJ intermolecular kernel (half-list)
function lj_kernel_mm!(i, F, Epbuf, Wbuf, pos, h, hi, nlc, nll)
    @inbounds begin
        fi0=0.0;fi1=0.0;fi2=0.0; my_Ep=0.0; ww0=0.0;ww4=0.0;ww8=0.0
        nni = nlc[i]
        for jn in 1:nni
            j = nll[(i-1)*MAX_LJ_NEIGH_MM+jn]
            dx=pos[3j-2]-pos[3i-2]; dy=pos[3j-1]-pos[3i-1]; dz=pos[3j]-pos[3i]
            s0=hi[1]*dx+hi[2]*dy+hi[3]*dz;s1=hi[4]*dx+hi[5]*dy+hi[6]*dz;s2=hi[7]*dx+hi[8]*dy+hi[9]*dz
            s0-=round(s0);s1-=round(s1);s2-=round(s2)
            dx=h[1]*s0+h[2]*s1+h[3]*s2;dy=h[4]*s0+h[5]*s1+h[6]*s2;dz=h[7]*s0+h[8]*s1+h[9]*s2
            r2=dx*dx+dy*dy+dz*dz
            r2 > RCUT2 && continue
            if r2 < 0.25; r2 = 0.25; end
            ri2=1.0/r2; sr2=sig2_LJ*ri2; sr6=sr2*sr2*sr2; sr12=sr6*sr6
            fm=24.0*eps_LJ*(2.0*sr12-sr6)*ri2
            my_Ep += 4.0*eps_LJ*(sr12-sr6)-VSHFT
            fi0 -= fm*dx; fi1 -= fm*dy; fi2 -= fm*dz
            @atomic F[3j-2] += fm*dx; @atomic F[3j-1] += fm*dy; @atomic F[3j] += fm*dz
            ww0 += dx*fm*dx; ww4 += dy*fm*dy; ww8 += dz*fm*dz
        end
        @atomic F[3i-2] += fi0; @atomic F[3i-1] += fi1; @atomic F[3i] += fi2
        Epbuf[i] = my_Ep
        Wbuf[3i-2] = ww0; Wbuf[3i-1] = ww4; Wbuf[3i] = ww8
    end; return nothing
end

# KE kernel
function ke_kernel_mm(i, vel)
    @inbounds mC * (vel[3i-2]^2 + vel[3i-1]^2 + vel[3i]^2)
end

# Velocity pre/post kernels
function vel_pre_mm_kernel!(i, vel, F, sc_v, hdt_mi)
    @inbounds begin
        vel[3i-2]=vel[3i-2]*sc_v+hdt_mi*F[3i-2]; vel[3i-1]=vel[3i-1]*sc_v+hdt_mi*F[3i-1]; vel[3i]=vel[3i]*sc_v+hdt_mi*F[3i]
    end; return nothing
end

function pos_update_mm_kernel!(i, pos, vel, hi, dt_val)
    @inbounds begin
        px=pos[3i-2];py=pos[3i-1];pz=pos[3i]; vx=vel[3i-2];vy=vel[3i-1];vz=vel[3i]
        sx=hi[1]*px+hi[2]*py+hi[3]*pz;sy=hi[4]*px+hi[5]*py+hi[6]*pz;sz=hi[7]*px+hi[8]*py+hi[9]*pz
        vsx=hi[1]*vx+hi[2]*vy+hi[3]*vz;vsy=hi[4]*vx+hi[5]*vy+hi[6]*vz;vsz=hi[7]*vx+hi[8]*vy+hi[9]*vz
        sx+=dt_val*vsx;sy+=dt_val*vsy;sz+=dt_val*vsz; sx-=floor(sx);sy-=floor(sy);sz-=floor(sz)
        pos[3i-2]=sx;pos[3i-1]=sy;pos[3i]=sz
    end; return nothing
end

function frac2cart_mm_kernel!(i, pos, h)
    @inbounds begin
        sx=pos[3i-2];sy=pos[3i-1];sz=pos[3i]
        pos[3i-2]=h[1]*sx+h[2]*sy+h[3]*sz; pos[3i-1]=h[4]*sx+h[5]*sy+h[6]*sz; pos[3i]=h[7]*sx+h[8]*sy+h[9]*sz
    end; return nothing
end

function vel_post_mm_kernel!(i, vel, F, sc_v2, hdt_mi)
    @inbounds begin
        vel[3i-2]=(vel[3i-2]+hdt_mi*F[3i-2])*sc_v2; vel[3i-1]=(vel[3i-1]+hdt_mi*F[3i-1])*sc_v2; vel[3i]=(vel[3i]+hdt_mi*F[3i])*sc_v2
    end; return nothing
end

function vel_scale_mm_kernel!(i, vel, scale)
    @inbounds begin; vel[3i-2]*=scale; vel[3i-1]*=scale; vel[3i]*=scale; end; return nothing
end

# ═══════════ High-level functions ═══════════

function compute_forces_mm!(F, vir9, Eb_ref, Ea_ref, Ed_ref, Ei_ref, Elj_ref,
        Epbuf_b, Wbuf_b, Epbuf_a, Epbuf_d, Epbuf_i, Epbuf_lj, Wbuf_lj,
        pos, h, hi, ft, nlc, nll, Na)
    JACC.parallel_for(Na, zero_kernel!, F)
    # Bond
    JACC.parallel_for(ft.Nb, bond_kernel!, F, Epbuf_b, Wbuf_b, pos, h, hi, ft.b_i0_d, ft.b_i1_d, ft.b_kb_d, ft.b_r0_d)
    # Angle
    JACC.parallel_for(ft.Nang, angle_kernel!, F, Epbuf_a, pos, h, hi, ft.ang_i0_d, ft.ang_i1_d, ft.ang_i2_d, ft.ang_kth_d, ft.ang_th0_d)
    # Dihedral
    JACC.parallel_for(ft.Ndih, dihedral_kernel!, F, Epbuf_d, pos, h, hi, ft.dih_i0_d, ft.dih_i1_d, ft.dih_i2_d, ft.dih_i3_d, ft.dih_Vn_d, ft.dih_mult_d, ft.dih_gamma_d)
    # Improper
    JACC.parallel_for(ft.Nimp, improper_kernel!, F, Epbuf_i, pos, h, hi, ft.imp_i0_d, ft.imp_i1_d, ft.imp_i2_d, ft.imp_i3_d, ft.imp_ki_d, ft.imp_gamma_d)
    # LJ intermolecular
    JACC.parallel_for(Na, lj_kernel_mm!, F, Epbuf_lj, Wbuf_lj, pos, h, hi, nlc, nll)

    # Host reductions
    Eb_ref[] = sum(Array(Epbuf_b))
    Ea_ref[] = sum(Array(Epbuf_a))
    Ed_ref[] = sum(Array(Epbuf_d))
    Ei_ref[] = sum(Array(Epbuf_i))
    Elj_ref[] = sum(Array(Epbuf_lj))
    Wbuf_b_h = Array(Wbuf_b); Wbuf_lj_h = Array(Wbuf_lj)
    fill!(vir9, 0.0)
    for i in 1:ft.Nb; vir9[1]+=Wbuf_b_h[3i-2]; vir9[5]+=Wbuf_b_h[3i-1]; vir9[9]+=Wbuf_b_h[3i]; end
    for i in 1:Na; vir9[1]+=Wbuf_lj_h[3i-2]; vir9[5]+=Wbuf_lj_h[3i-1]; vir9[9]+=Wbuf_lj_h[3i]; end
    return nothing
end

function step_npt_mm!(pos, vel, F, vir9, h, hi, Na, dt, npt, ft,
        Epbuf_b, Wbuf_b, Epbuf_a, Epbuf_d, Epbuf_i, Epbuf_lj, Wbuf_lj,
        nlc, nll)
    hdt = 0.5*dt
    V = abs(mat_det9(h))
    KE = 0.5 * JACC.parallel_reduce(Na, ke_kernel_mm, vel) / CONV
    npt.xi += hdt*(2*KE - npt.Nf*kB*npt.Tt)/npt.Q; npt.xi = clamp(npt.xi, -0.05, 0.05)
    dP = inst_P(vir9, KE, V) - npt.Pe
    for a in 0:2; npt.Vg[a*4+1] += hdt*V*dP/(npt.W*eV2GPa); npt.Vg[a*4+1] = clamp(npt.Vg[a*4+1], -0.005, 0.005); end
    h_h = Array(h); hi_h = Array(hi)
    eps_tr = npt.Vg[1]*hi_h[1]+npt.Vg[5]*hi_h[5]+npt.Vg[9]*hi_h[9]
    sc_nh = exp(-hdt*npt.xi); sc_v = sc_nh*exp(-hdt*eps_tr/3.0)
    hdt_mi = hdt*CONV/mC
    JACC.parallel_for(Na, vel_pre_mm_kernel!, vel, F, sc_v, hdt_mi)
    JACC.parallel_for(Na, pos_update_mm_kernel!, pos, vel, hi, dt)
    for a in 0:2, b in 0:2; h_h[a*3+b+1] += dt*npt.Vg[a*3+b+1]; end
    mat_inv9!(hi_h, h_h)
    copyto!(h, JACC.array(h_h)); copyto!(hi, JACC.array(hi_h))
    JACC.parallel_for(Na, frac2cart_mm_kernel!, pos, h)

    Eb_ref=Ref(0.0); Ea_ref=Ref(0.0); Ed_ref=Ref(0.0); Ei_ref=Ref(0.0); Elj_ref=Ref(0.0)
    compute_forces_mm!(F, vir9, Eb_ref, Ea_ref, Ed_ref, Ei_ref, Elj_ref,
        Epbuf_b, Wbuf_b, Epbuf_a, Epbuf_d, Epbuf_i, Epbuf_lj, Wbuf_lj,
        pos, h, hi, ft, nlc, nll, Na)

    eps_tr2 = npt.Vg[1]*hi_h[1]+npt.Vg[5]*hi_h[5]+npt.Vg[9]*hi_h[9]
    sc_v2 = sc_nh*exp(-hdt*eps_tr2/3.0)
    JACC.parallel_for(Na, vel_post_mm_kernel!, vel, F, sc_v2, hdt_mi)
    KE = 0.5 * JACC.parallel_reduce(Na, ke_kernel_mm, vel) / CONV
    npt.xi += hdt*(2*KE - npt.Nf*kB*npt.Tt)/npt.Q; npt.xi = clamp(npt.xi, -0.05, 0.05)
    V2 = abs(mat_det9(h_h))
    dP = inst_P(vir9, KE, V2) - npt.Pe
    for a in 0:2; npt.Vg[a*4+1] += hdt*V2*dP/(npt.W*eV2GPa); npt.Vg[a*4+1] = clamp(npt.Vg[a*4+1], -0.005, 0.005); end
    return (Eb=Eb_ref[], Ea=Ea_ref[], Ed=Ed_ref[], Ei=Ei_ref[], Elj=Elj_ref[], KE=KE)
end

# ═══════════ Restart I/O ═══════════
function write_restart_mmmd(fname, istep, opts, st, nc, T, Pe, nsteps, dt, seed, fspec, init_scale,
                            h, npt, pos, vel, Na, Nmol, natom)
    open(fname, "w") do f
        print(f, "# RESTART fuller_LJ_npt_mmmd_julia\n# OPTIONS:")
        for (k,v) in opts; print(f, " --$k=$v"); end
        @printf(f, "\nSTEP %d\nNSTEPS %d\nDT %.15e\nTEMP %.15e\nPRES %.15e\n", istep, nsteps, dt, T, Pe)
        @printf(f, "CRYSTAL %s\nNC %d\nFULLERENE %s\nINIT_SCALE %.15e\nSEED %d\n", st, nc, fspec, init_scale, seed)
        @printf(f, "NMOL %d\nNATOM_MOL %d\nNATOM %d\n", Nmol, natom, Na)
        print(f, "H"); for i in 1:9; @printf(f, " %.15e", h[i]); end; print(f, "\n")
        @printf(f, "NPT %.15e %.15e %.15e %.15e %.15e %d\n", npt.xi, npt.Q, npt.W, npt.Pe, npt.Tt, npt.Nf)
        print(f, "VG"); for i in 1:9; @printf(f, " %.15e", npt.Vg[i]); end; print(f, "\n")
        for i in 1:Na
            @printf(f, "ATOM %d %.15e %.15e %.15e %.15e %.15e %.15e\n",
                i, pos[3i-2],pos[3i-1],pos[3i], vel[3i-2],vel[3i-1],vel[3i])
        end
        print(f, "END\n")
    end
end

function read_restart_mmmd(fname)
    h = zeros(9); Vg = zeros(9)
    xi=0.0; Q=0.0; W=0.0; Pe_r=0.0; Tt_r=0.0; Nf=0; istep=0
    pos_v=Float64[]; vel_v=Float64[]; ok = false
    for line in readlines(fname)
        line = strip(line)
        if isempty(line) || line[1]=='#'; continue; end
        parts = split(line); tag = parts[1]
        if tag=="STEP"; istep=parse(Int,parts[2])
        elseif tag=="H"; for i in 1:9; h[i]=parse(Float64,parts[i+1]); end
        elseif tag=="NPT"
            xi=parse(Float64,parts[2]); Q=parse(Float64,parts[3]); W=parse(Float64,parts[4])
            Pe_r=parse(Float64,parts[5]); Tt_r=parse(Float64,parts[6]); Nf=parse(Int,parts[7])
        elseif tag=="VG"; for i in 1:9; Vg[i]=parse(Float64,parts[i+1]); end
        elseif tag=="ATOM"
            px=parse(Float64,parts[3]);py=parse(Float64,parts[4]);pz=parse(Float64,parts[5])
            vx=parse(Float64,parts[6]);vy=parse(Float64,parts[7]);vz=parse(Float64,parts[8])
            append!(pos_v, [px,py,pz]); append!(vel_v, [vx,vy,vz])
        elseif tag=="END"; break; end
    end
    Na_r = div(length(pos_v), 3); ok = Na_r > 0
    npt = NPTState(xi, Q, Vg, W, Pe_r, Tt_r, Nf)
    if ok; @printf("  Restart loaded: %s (step %d, %d atoms)\n", fname, istep, Na_r); end
    return (istep=istep, Na=Na_r, h=h, npt=npt, pos=pos_v, vel=vel_v, ok=ok)
end

# ═══════════ MAIN ═══════════
function main()
    opts = parse_args(ARGS)
    if haskey(opts, "help")
        println("""fuller_LJ_npt_mmmd.jl — MM+LJ fullerene NPT-MD (JACC.jl)

Options:
  --help                  Show this help
  --fullerene=<name>      Fullerene species (default: C60)
  --crystal=<fcc|hcp|bcc> Crystal structure (default: fcc)
  --cell=<nc>             Unit cell repeats (default: 3)
  --temp=<K>              Target temperature [K] (default: 298.0)
  --pres=<GPa>            Target pressure [GPa] (default: 0.0)
  --step=<N>              Production steps (default: 10000)
  --dt=<fs>               Time step [fs] (default: 0.1)
  --init_scale=<s>        Lattice scale factor (default: 1.0)
  --seed=<n>              Random seed (default: 42)
  --coldstart=<N>         Cold-start steps at 4K (default: 0)
  --warmup=<N>            Warm-up ramp steps 4K->T (default: 0)
  --from=<step>           Averaging start step (default: auto)
  --to=<step>             Averaging end step (default: nsteps)
  --mon=<N>               Monitor print interval (default: auto)
  --warmup_mon=<mode>     Warmup monitor: norm|freq|some (default: norm)
  --ovito=<N>             OVITO XYZ output interval, 0=off (default: 0)
  --ofile=<filename>      OVITO output filename (default: auto)
  --restart=<N>           Restart save interval, 0=off (default: 0)
  --resfile=<path>        Resume from restart file
  --libdir=<path>         Fullerene library dir (default: FullereneLib)
  --ff_kb=<kcal/mol>      Bond stretch constant (default: 469.0)
  --ff_kth=<kcal/mol>     Angle bend constant (default: 63.0)
  --ff_v2=<kcal/mol>      Dihedral constant (default: 14.5)
  --ff_kimp=<kcal/mol>    Improper constant (default: 15.0)""")
        return 0
    end

    crystal = get_opt(opts, "crystal", "fcc"); st = uppercase(crystal)
    nc = parse(Int, get_opt(opts, "cell", "3"))
    T = parse(Float64, get_opt(opts, "temp", "298.0"))
    Pe = parse(Float64, get_opt(opts, "pres", "0.0"))
    nsteps = parse(Int, get_opt(opts, "step", "10000"))
    dt = parse(Float64, get_opt(opts, "dt", "0.1"))
    seed = parse(Int, get_opt(opts, "seed", "42"))
    init_scale = parse(Float64, get_opt(opts, "init_scale", "1.0"))
    fspec = get_opt(opts, "fullerene", "C60"); libdir = get_opt(opts, "libdir", "FullereneLib")
    coldstart = parse(Int, get_opt(opts, "coldstart", "0"))
    warmup = parse(Int, get_opt(opts, "warmup", "0"))
    avg_from = parse(Int, get_opt(opts, "from", "0")); avg_to = parse(Int, get_opt(opts, "to", "0"))
    nrec_o = parse(Int, get_opt(opts, "ovito", "0"))
    nrec_rst = parse(Int, get_opt(opts, "restart", "0"))
    resfile = get_opt(opts, "resfile", "")
    mon_interval = parse(Int, get_opt(opts, "mon", "0"))
    warmup_mon_mode = get_opt(opts, "warmup_mon", "norm"); T_cold = 4.0

    ff_kb = parse(Float64, get_opt(opts, "ff_kb", "469.0")) * kcal2eV
    ff_kth = parse(Float64, get_opt(opts, "ff_kth", "63.0")) * kcal2eV
    ff_v2 = parse(Float64, get_opt(opts, "ff_v2", "14.5")) * kcal2eV
    ff_kimp = parse(Float64, get_opt(opts, "ff_kimp", "15.0")) * kcal2eV

    if avg_to <= 0; avg_to = nsteps; end
    if avg_from <= 0; avg_from = max(1, nsteps - div(nsteps, 4)); end
    total_steps = coldstart + warmup + nsteps
    gavg_from = coldstart + warmup + avg_from; gavg_to = coldstart + warmup + avg_to

    sfx = build_suffix(opts); mode_tag = "julia"
    ovito_file = haskey(opts, "ofile") ? get_opt(opts, "ofile", "") :
                 unique_file("ovito_traj_mmmd_" * mode_tag * sfx, ".xyz")

    fpath, label = resolve_fullerene(fspec, libdir)
    mol = load_cc1(fpath); natom = mol.natom
    a0 = default_a0(mol.Dmol > 0 ? mol.Dmol : 2*mol.Rmol, st, init_scale)
    Nmol_max = st=="FCC" ? 4*nc^3 : 2*nc^3

    # Build crystal
    mol_centers = zeros(Nmol_max*3); h = zeros(9); hi = zeros(9)
    if st=="FCC"; Nmol = make_fcc!(mol_centers, h, a0, nc)
    elseif st=="HCP"; Nmol = make_hcp!(mol_centers, h, a0, nc)
    else; Nmol = make_bcc!(mol_centers, h, a0, nc); end
    Na = Nmol * natom; mat_inv9!(hi, h)

    # Build all-atom positions
    pos_h = zeros(Na*3); vel_h = zeros(Na*3)
    for m in 1:Nmol, a in 1:natom
        idx = (m-1)*natom + a
        pos_h[3idx-2] = mol_centers[3m-2]+mol.coords[3a-2]; pos_h[3idx-1] = mol_centers[3m-1]+mol.coords[3a-1]; pos_h[3idx] = mol_centers[3m]+mol.coords[3a]
    end

    # Build topology
    ft_h = build_flat_topology(mol, Nmol, ff_kb, ff_kth, ff_v2, ff_kimp)
    mol_id_h = ft_h.mol_id

    @printf("========================================================================\n")
    @printf("  Fullerene Crystal NPT-MD — MM Force Field (JACC.jl)\n")
    @printf("========================================================================\n")
    @printf("  Fullerene       : %s (%d atoms/mol)\n", label, natom)
    @printf("  Crystal         : %s %dx%dx%d  Nmol=%d  Natom=%d\n", st, nc, nc, nc, Nmol, Na)
    @printf("  a0=%.3f  T=%.1f K  P=%.4f GPa  dt=%.3f fs\n", a0, T, Pe, dt)
    @printf("  Production      : %d steps  Total=%d\n", nsteps, total_steps)
    @printf("========================================================================\n\n")

    # Initial velocities
    T_init = (coldstart>0 || warmup>0) ? T_cold : T
    rng = MersenneTwister(seed); sv = sqrt(kB*T_init*CONV/mC)
    for i in 1:Na, a in 1:3; vel_h[3(i-1)+a] = sv*randn(rng); end
    vcm = zeros(3); for i in 1:Na, a in 1:3; vcm[a] += vel_h[3(i-1)+a]; end; vcm ./= Na
    for i in 1:Na, a in 1:3; vel_h[3(i-1)+a] -= vcm[a]; end

    Nf = 3*Na-3
    npt = NPTState(0.0, max(Nf*kB*T*1e4, 1e-20), zeros(9), max((Nf+9)*kB*T*1e6, 1e-20), Pe, T_init, Nf)
    start_step = 0

    if !isempty(resfile)
        rd = read_restart_mmmd(resfile)
        if rd.ok
            start_step = rd.istep; h .= rd.h; mat_inv9!(hi, h); npt = rd.npt
            nn = min(Na*3, rd.Na*3)
            pos_h[1:nn] .= rd.pos[1:nn]; vel_h[1:nn] .= rd.vel[1:nn]
            @printf("  Restarting from global step %d\n", start_step)
        end
    end

    # Transfer to device
    pos = JACC.array(pos_h); vel = JACC.array(vel_h); F = JACC.zeros(Float64, Na*3)
    h_d = JACC.array(h); hi_d = JACC.array(hi); vir9 = zeros(9)

    # Topology arrays to device
    ft_d = (Nb=ft_h.Nb, Nang=ft_h.Nang, Ndih=ft_h.Ndih, Nimp=ft_h.Nimp,
            b_i0_d=JACC.array(ft_h.b_i0), b_i1_d=JACC.array(ft_h.b_i1), b_kb_d=JACC.array(ft_h.b_kb), b_r0_d=JACC.array(ft_h.b_r0),
            ang_i0_d=JACC.array(ft_h.ang_i0), ang_i1_d=JACC.array(ft_h.ang_i1), ang_i2_d=JACC.array(ft_h.ang_i2),
            ang_kth_d=JACC.array(ft_h.ang_kth), ang_th0_d=JACC.array(ft_h.ang_th0),
            dih_i0_d=JACC.array(ft_h.dih_i0), dih_i1_d=JACC.array(ft_h.dih_i1), dih_i2_d=JACC.array(ft_h.dih_i2), dih_i3_d=JACC.array(ft_h.dih_i3),
            dih_Vn_d=JACC.array(ft_h.dih_Vn), dih_mult_d=JACC.array(ft_h.dih_mult), dih_gamma_d=JACC.array(ft_h.dih_gamma),
            imp_i0_d=JACC.array(ft_h.imp_i0), imp_i1_d=JACC.array(ft_h.imp_i1), imp_i2_d=JACC.array(ft_h.imp_i2), imp_i3_d=JACC.array(ft_h.imp_i3),
            imp_ki_d=JACC.array(ft_h.imp_ki), imp_gamma_d=JACC.array(ft_h.imp_gamma))

    # Energy/virial buffers
    Epbuf_b = JACC.zeros(Float64, ft_h.Nb); Wbuf_b = JACC.zeros(Float64, ft_h.Nb*3)
    Epbuf_a = JACC.zeros(Float64, ft_h.Nang); Epbuf_d = JACC.zeros(Float64, ft_h.Ndih)
    Epbuf_i = JACC.zeros(Float64, ft_h.Nimp)
    Epbuf_lj = JACC.zeros(Float64, Na); Wbuf_lj = JACC.zeros(Float64, Na*3)

    # LJ neighbor list
    nlc_h = zeros(Int32, Na); nll_h = zeros(Int32, Na*MAX_LJ_NEIGH_MM)
    build_nlist_lj_mm!(nlc_h, nll_h, pos_h, h, hi, Na, mol_id_h)
    nlc_d = JACC.array(nlc_h); nll_d = JACC.array(nll_h)

    # Initial PBC + forces
    JACC.parallel_for(Na, pbc_kernel!, pos, h_d, hi_d)
    Eb_ref=Ref(0.0); Ea_ref=Ref(0.0); Ed_ref=Ref(0.0); Ei_ref=Ref(0.0); Elj_ref=Ref(0.0)
    compute_forces_mm!(F, vir9, Eb_ref, Ea_ref, Ed_ref, Ei_ref, Elj_ref,
        Epbuf_b, Wbuf_b, Epbuf_a, Epbuf_d, Epbuf_i, Epbuf_lj, Wbuf_lj,
        pos, h_d, hi_d, ft_d, nlc_d, nll_d, Na)

    prn = mon_interval>0 ? mon_interval : max(1, div(total_steps, 50))
    prn_pre = prn
    if coldstart+warmup>0
        div_val = warmup_mon_mode=="freq" ? 10 : warmup_mon_mode=="some" ? 1000 : 100
        prn_pre = max(1, div(coldstart+warmup, div_val))
    end
    nlup = 20
    sT=0.0;sP=0.0;sa=0.0;sEb=0.0;sEa=0.0;sEd=0.0;sElj=0.0;sEt=0.0;nav=0
    t0 = time()
    io_o = nrec_o>0 ? open(ovito_file, "w") : nothing
    rst_base = nrec_o>0 ? ovito_file : ("restart_mmmd_" * mode_tag)

    @printf("  %8s %5s %7s %9s %8s %9s %9s %9s %9s %9s %7s\n",
        "step","phase","T[K]","P[GPa]","a[A]","E_bond","E_angle","E_dih","E_LJ","E_total","t[s]")

    stop_requested = false
    for gstep in (start_step+1):total_steps
        phase = gstep<=coldstart ? "COLD" : gstep<=coldstart+warmup ? "WARM" : "PROD"
        cur_prn = gstep<=coldstart+warmup ? prn_pre : prn

        if gstep<=coldstart; npt.Tt=T_cold
        elseif gstep<=coldstart+warmup
            npt.Tt=T_cold+(T-T_cold)*Float64(gstep-coldstart)/Float64(warmup)
        else; npt.Tt=T; end
        if coldstart>0 && gstep==coldstart+1; npt.xi=0.0; fill!(npt.Vg,0.0); end
        if gstep<=coldstart; fill!(npt.Vg,0.0); end

        if gstep%nlup==0
            pos_h .= Array(pos); mat_inv9!(hi, h)
            build_nlist_lj_mm!(nlc_h, nll_h, pos_h, h, hi, Na, mol_id_h)
            nlc_d = JACC.array(nlc_h); nll_d = JACC.array(nll_h)
            h_d = JACC.array(h); hi_d = JACC.array(hi)
        end

        fr = step_npt_mm!(pos, vel, F, vir9, h_d, hi_d, Na, dt, npt, ft_d,
            Epbuf_b, Wbuf_b, Epbuf_a, Epbuf_d, Epbuf_i, Epbuf_lj, Wbuf_lj,
            nlc_d, nll_d)
        h .= Array(h_d); hi .= Array(hi_d)
        Etot = fr.Eb + fr.Ea + fr.Ed + fr.Ei + fr.Elj

        V = abs(mat_det9(h)); Tn = inst_T(fr.KE, npt.Nf); Pn = inst_P(vir9, fr.KE, V)

        if (gstep<=coldstart || gstep<=coldstart+warmup) && Tn>0.1
            tgt = gstep<=coldstart ? T_cold : npt.Tt
            scale = sqrt(max(tgt, 0.1)/Tn)
            JACC.parallel_for(Na, vel_scale_mm_kernel!, vel, scale)
            KE2 = 0.5*JACC.parallel_reduce(Na, ke_kernel_mm, vel)/CONV; Tn = inst_T(KE2, npt.Nf)
            npt.xi = 0.0; if gstep<=coldstart; fill!(npt.Vg,0.0); end
        end

        an = h[1]/nc
        if gstep>=gavg_from && gstep<=gavg_to
            sT+=Tn;sP+=Pn;sa+=an;sEb+=fr.Eb/Nmol;sEa+=fr.Ea/Nmol;sEd+=(fr.Ed+fr.Ei)/Nmol;sElj+=fr.Elj/Nmol;sEt+=Etot/Nmol;nav+=1
        end

        if io_o!==nothing && gstep%nrec_o==0
            pos_h_o=Array(pos); vel_h_o=Array(vel)
            write_ovito_allatom(io_o, gstep, dt, pos_h_o, vel_h_o, h, Na, mol_id_h)
            flush(io_o)
        end

        if nrec_rst>0 && (gstep%nrec_rst==0 || gstep==total_steps)
            rfn = restart_filename(rst_base, gstep, total_steps)
            pos_h_r=Array(pos); vel_h_r=Array(vel)
            write_restart_mmmd(rfn, gstep, opts, st, nc, T, Pe, nsteps, dt, seed, fspec, init_scale,
                h, npt, pos_h_r, vel_h_r, Na, Nmol, natom)
            if stop_requested
                @printf("\n  *** Stopped at restart checkpoint (step %d) ***\n", gstep); break
            end
        end

        if gstep%cur_prn==0 || gstep==total_steps
            if dir_exists("abort.md")
                @printf("\n  *** abort.md detected at step %d ***\n", gstep)
                if nrec_rst>0
                    rfn=restart_filename(rst_base,gstep,total_steps); pos_h_r=Array(pos); vel_h_r=Array(vel)
                    write_restart_mmmd(rfn,gstep,opts,st,nc,T,Pe,nsteps,dt,seed,fspec,init_scale,h,npt,pos_h_r,vel_h_r,Na,Nmol,natom)
                end; break
            end
            if !stop_requested && dir_exists("stop.md")
                stop_requested=true
                @printf("\n  *** stop.md detected at step %d — will stop at next checkpoint ***\n", gstep)
                if nrec_rst==0; break; end
            end
            el = time()-t0
            @printf("  %8d %5s %7.1f %9.3f %8.3f %9.4f %9.4f %9.4f %9.4f %9.4f %7.0f\n",
                gstep,phase,Tn,Pn,an,fr.Eb/Nmol,fr.Ea/Nmol,(fr.Ed+fr.Ei)/Nmol,fr.Elj/Nmol,Etot/Nmol,el)
        end
    end

    if io_o!==nothing; close(io_o); end
    if nav>0
        @printf("\n========================================================================\n")
        @printf("  Averages (%d): T=%.2f P=%.4f a=%.4f  bond=%.4f ang=%.4f dih=%.4f LJ=%.4f tot=%.4f\n",
            nav,sT/nav,sP/nav,sa/nav,sEb/nav,sEa/nav,sEd/nav,sElj/nav,sEt/nav)
        @printf("========================================================================\n")
    end
    @printf("  Done (%.1f sec)\n", time()-t0)
    return 0
end

main()
