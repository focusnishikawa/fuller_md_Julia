#=============================================================================
  fuller_airebo_npt_md.jl — Fullerene Crystal NPT-MD (AIREBO: REBO-II + LJ)
  Portable CPU/GPU version using JACC.jl

  Usage:
    julia --project=.. fuller_airebo_npt_md.jl [options]

  Options:
    --help                  Show help
    --fullerene=<name>      Fullerene species (default: C60)
    --crystal=<fcc|hcp|bcc> Crystal structure (default: fcc)
    --cell=<nc>             Unit cell repeats (default: 3)
    --temp=<K>              Target temperature [K] (default: 298.0)
    --pres=<GPa>            Target pressure [GPa] (default: 0.0)
    --step=<N>              Production steps (default: 10000)
    --dt=<fs>               Time step [fs] (default: 0.5)
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

  Stop control: create abort.md/ or stop.md/ directory during execution.
  Unit system: A, amu, eV, fs, K, GPa
=============================================================================#

using JACC
using Printf, Random
JACC.@init_backend

include("FullerMD.jl")
using .FullerMD

# ═══════════ REBO-II C-C Parameters ═══════════
const Q_CC = 0.3134602960833
const A_CC = 10953.544162170
const alpha_CC = 4.7465390606595
const B1_CC = 12388.79197798; const beta1_CC = 4.7204523127
const B2_CC = 17.56740646509; const beta2_CC = 1.4332132499
const B3_CC = 30.71493208065; const beta3_CC = 1.3826912506
const Dmin_CC = 1.7; const Dmax_CC = 2.0
const BO_DELTA = 0.5
const G_a0 = 0.00020813; const G_c0 = 330.0; const G_d0 = 3.5
const REBO_RCUT = Dmax_CC + 0.3
const MAX_REBO_NEIGH = 12

# ═══════════ LJ intermolecular Parameters ═══════════
const sig_LJ_a = 3.40; const eps_LJ_a = 2.84e-3
const LJ_RCUT_a = 3.0 * sig_LJ_a
const LJ_RCUT2_a = LJ_RCUT_a * LJ_RCUT_a
const sig2_LJ_a = sig_LJ_a * sig_LJ_a
const _sr_v = sig_LJ_a / LJ_RCUT_a
const _sr2 = _sr_v * _sr_v; const _sr6 = _sr2 * _sr2 * _sr2
const LJ_VSHFT_a = 4.0 * eps_LJ_a * (_sr6 * _sr6 - _sr6)
const MAX_LJ_NEIGH = 400

# ═══════════ REBO device helper functions ═══════════
@inline function fc_d(r, dmin, dmax)
    r <= dmin && return 1.0
    r >= dmax && return 0.0
    return 0.5 * (1.0 + cos(π * (r - dmin) / (dmax - dmin)))
end

@inline function dfc_d(r, dmin, dmax)
    (r <= dmin || r >= dmax) && return 0.0
    return -0.5 * π / (dmax - dmin) * sin(π * (r - dmin) / (dmax - dmin))
end

@inline VR_CC_d(r) = (1.0 + Q_CC / r) * A_CC * exp(-alpha_CC * r)

@inline function dVR_CC_d(r)
    ex = A_CC * exp(-alpha_CC * r)
    return (-Q_CC / (r * r)) * ex + (1.0 + Q_CC / r) * (-alpha_CC) * ex
end

@inline VA_CC_d(r) = B1_CC * exp(-beta1_CC * r) + B2_CC * exp(-beta2_CC * r) + B3_CC * exp(-beta3_CC * r)

@inline function dVA_CC_d(r)
    return -beta1_CC * B1_CC * exp(-beta1_CC * r) -
            beta2_CC * B2_CC * exp(-beta2_CC * r) -
            beta3_CC * B3_CC * exp(-beta3_CC * r)
end

@inline function G_C_d(x)
    c2 = G_c0 * G_c0; d2 = G_d0 * G_d0; hv = 1.0 + x
    return G_a0 * (1.0 + c2 / d2 - c2 / (d2 + hv * hv))
end

@inline function dG_C_d(x)
    c2 = G_c0 * G_c0; d2 = G_d0 * G_d0; hv = 1.0 + x; dn = d2 + hv * hv
    return G_a0 * 2.0 * c2 * hv / (dn * dn)
end

# ═══════════ Neighbor list builders (host) ═══════════
function build_nlist_rebo!(nlc, nll, pos, h, hi, Na, mol_id)
    rc2 = REBO_RCUT * REBO_RCUT
    fill!(nlc, 0)
    for i in 1:Na-1
        mi = mol_id[i]
        for j in i+1:Na
            mol_id[j] != mi && continue
            dx = pos[3j-2] - pos[3i-2]; dy = pos[3j-1] - pos[3i-1]; dz = pos[3j] - pos[3i]
            dx, dy, dz = mimg(dx, dy, dz, hi, h)
            dx*dx + dy*dy + dz*dz >= rc2 && continue
            ci = nlc[i] + 1
            if ci <= MAX_REBO_NEIGH; nll[(i-1)*MAX_REBO_NEIGH+ci] = j; nlc[i] = ci; end
            cj = nlc[j] + 1
            if cj <= MAX_REBO_NEIGH; nll[(j-1)*MAX_REBO_NEIGH+cj] = i; nlc[j] = cj; end
        end
    end
end

function build_nlist_lj!(nlc, nll, pos, h, hi, Na, mol_id)
    rc2 = (LJ_RCUT_a + 2.0)^2
    fill!(nlc, 0)
    for i in 1:Na-1
        mi = mol_id[i]
        for j in i+1:Na
            mol_id[j] == mi && continue
            dx = pos[3j-2] - pos[3i-2]; dy = pos[3j-1] - pos[3i-1]; dz = pos[3j] - pos[3i]
            dx, dy, dz = mimg(dx, dy, dz, hi, h)
            dx*dx + dy*dy + dz*dz >= rc2 && continue
            ci = nlc[i] + 1
            if ci <= MAX_LJ_NEIGH; nll[(i-1)*MAX_LJ_NEIGH+ci] = j; nlc[i] = ci; end
        end
    end
end

# ═══════════ JACC Kernels ═══════════

# Zero force array
function zero_kernel!(i, F)
    @inbounds begin; F[3i-2] = 0.0; F[3i-1] = 0.0; F[3i] = 0.0; end
    return nothing
end

# PBC kernel
function pbc_kernel!(i, pos, h, hi)
    @inbounds begin
        px = pos[3i-2]; py = pos[3i-1]; pz = pos[3i]
        s0 = hi[1]*px+hi[2]*py+hi[3]*pz; s1 = hi[4]*px+hi[5]*py+hi[6]*pz; s2 = hi[7]*px+hi[8]*py+hi[9]*pz
        s0 -= floor(s0); s1 -= floor(s1); s2 -= floor(s2)
        pos[3i-2] = h[1]*s0+h[2]*s1+h[3]*s2; pos[3i-1] = h[4]*s0+h[5]*s1+h[6]*s2; pos[3i] = h[7]*s0+h[8]*s1+h[9]*s2
    end; return nothing
end

# REBO-II force kernel (half-list j>i, @atomic for F[j]/F[k]/F[l])
function rebo_kernel!(i, F, Epbuf, Wbuf, pos, h, hi, nlc, nll)
    @inbounds begin
        fi0 = 0.0; fi1 = 0.0; fi2 = 0.0; my_Ep = 0.0
        w00=0.0;w01=0.0;w02=0.0;w10=0.0;w11=0.0;w12=0.0;w20=0.0;w21=0.0;w22=0.0
        nni = nlc[i]

        for jn in 1:nni
            j = nll[(i-1)*MAX_REBO_NEIGH+jn]
            j <= i && continue  # half-list: only j > i
            dx = pos[3j-2]-pos[3i-2]; dy = pos[3j-1]-pos[3i-1]; dz = pos[3j]-pos[3i]
            s0 = hi[1]*dx+hi[2]*dy+hi[3]*dz; s1 = hi[4]*dx+hi[5]*dy+hi[6]*dz; s2 = hi[7]*dx+hi[8]*dy+hi[9]*dz
            s0 -= round(s0); s1 -= round(s1); s2 -= round(s2)
            dx = h[1]*s0+h[2]*s1+h[3]*s2; dy = h[4]*s0+h[5]*s1+h[6]*s2; dz = h[7]*s0+h[8]*s1+h[9]*s2
            rij = sqrt(dx*dx+dy*dy+dz*dz)
            rij > Dmax_CC && continue
            fcut = fc_d(rij, Dmin_CC, Dmax_CC); dfcut = dfc_d(rij, Dmin_CC, Dmax_CC)
            (fcut < 1e-15 && dfcut == 0.0) && continue
            vr = VR_CC_d(rij); dvr = dVR_CC_d(rij); va = VA_CC_d(rij); dva = dVA_CC_d(rij)
            rij_inv = 1.0 / rij
            rhat0 = dx*rij_inv; rhat1 = dy*rij_inv; rhat2 = dz*rij_inv

            # b_ij: angular screening over k-neighbors of i
            Gs_ij = 0.0
            for kn in 1:nni
                k = nll[(i-1)*MAX_REBO_NEIGH+kn]; k == j && continue
                dkx = pos[3k-2]-pos[3i-2]; dky = pos[3k-1]-pos[3i-1]; dkz = pos[3k]-pos[3i]
                sk0 = hi[1]*dkx+hi[2]*dky+hi[3]*dkz; sk1 = hi[4]*dkx+hi[5]*dky+hi[6]*dkz; sk2 = hi[7]*dkx+hi[8]*dky+hi[9]*dkz
                sk0 -= round(sk0); sk1 -= round(sk1); sk2 -= round(sk2)
                dkx = h[1]*sk0+h[2]*sk1+h[3]*sk2; dky = h[4]*sk0+h[5]*sk1+h[6]*sk2; dkz = h[7]*sk0+h[8]*sk1+h[9]*sk2
                rik = sqrt(dkx*dkx+dky*dky+dkz*dkz)
                rik > Dmax_CC && continue
                fc_ik = fc_d(rik, Dmin_CC, Dmax_CC); fc_ik < 1e-15 && continue
                costh = (dx*dkx+dy*dky+dz*dkz) / (rij*rik)
                costh = clamp(costh, -1.0, 1.0)
                Gs_ij += fc_ik * G_C_d(costh)
            end
            bij = (1.0 + Gs_ij)^(-BO_DELTA)

            # b_ji: angular screening over l-neighbors of j
            Gs_ji = 0.0
            nnj = nlc[j]
            for ln in 1:nnj
                l = nll[(j-1)*MAX_REBO_NEIGH+ln]; l == i && continue
                dlx = pos[3l-2]-pos[3j-2]; dly = pos[3l-1]-pos[3j-1]; dlz = pos[3l]-pos[3j]
                sl0 = hi[1]*dlx+hi[2]*dly+hi[3]*dlz; sl1 = hi[4]*dlx+hi[5]*dly+hi[6]*dlz; sl2 = hi[7]*dlx+hi[8]*dly+hi[9]*dlz
                sl0 -= round(sl0); sl1 -= round(sl1); sl2 -= round(sl2)
                dlx = h[1]*sl0+h[2]*sl1+h[3]*sl2; dly = h[4]*sl0+h[5]*sl1+h[6]*sl2; dlz = h[7]*sl0+h[8]*sl1+h[9]*sl2
                rjl = sqrt(dlx*dlx+dly*dly+dlz*dlz)
                rjl > Dmax_CC && continue
                fc_jl = fc_d(rjl, Dmin_CC, Dmax_CC); fc_jl < 1e-15 && continue
                costh = (-dx*dlx-dy*dly-dz*dlz) / (rij*rjl)
                costh = clamp(costh, -1.0, 1.0)
                Gs_ji += fc_jl * G_C_d(costh)
            end
            bji = (1.0 + Gs_ji)^(-BO_DELTA)
            bbar = 0.5 * (bij + bji)

            my_Ep += fcut * (vr - bbar * va)
            fpair = (dfcut*(vr-bbar*va) + fcut*(dvr-bbar*dva)) * rij_inv

            # Pair force: F[i] local, F[j] atomic
            fi0 += fpair*dx; fi1 += fpair*dy; fi2 += fpair*dz
            @atomic F[3j-2] -= fpair*dx
            @atomic F[3j-1] -= fpair*dy
            @atomic F[3j]   -= fpair*dz

            # Pair virial (per-atom buffer, no atomics needed)
            w00 -= dx*fpair*dx; w01 -= dx*fpair*dy; w02 -= dx*fpair*dz
            w10 -= dy*fpair*dx; w11 -= dy*fpair*dy; w12 -= dy*fpair*dz
            w20 -= dz*fpair*dx; w21 -= dz*fpair*dy; w22 -= dz*fpair*dz

            # 3-body: db_ij/dr_k
            if abs(Gs_ij) > 1e-20 && va > 1e-20
                dbp = -BO_DELTA * (1.0+Gs_ij)^(-BO_DELTA-1.0)
                vh = 0.5 * fcut * va
                for kn in 1:nni
                    k = nll[(i-1)*MAX_REBO_NEIGH+kn]; k == j && continue
                    dkx = pos[3k-2]-pos[3i-2]; dky = pos[3k-1]-pos[3i-1]; dkz = pos[3k]-pos[3i]
                    sk0 = hi[1]*dkx+hi[2]*dky+hi[3]*dkz; sk1 = hi[4]*dkx+hi[5]*dky+hi[6]*dkz; sk2 = hi[7]*dkx+hi[8]*dky+hi[9]*dkz
                    sk0 -= round(sk0); sk1 -= round(sk1); sk2 -= round(sk2)
                    dkx = h[1]*sk0+h[2]*sk1+h[3]*sk2; dky = h[4]*sk0+h[5]*sk1+h[6]*sk2; dkz = h[7]*sk0+h[8]*sk1+h[9]*sk2
                    rik = sqrt(dkx*dkx+dky*dky+dkz*dkz)
                    rik > Dmax_CC && continue
                    fc_ik = fc_d(rik, Dmin_CC, Dmax_CC); fc_ik < 1e-15 && continue
                    dfc_ik = dfc_d(rik, Dmin_CC, Dmax_CC)
                    rik_inv = 1.0 / rik
                    rhat_ik0 = dkx*rik_inv; rhat_ik1 = dky*rik_inv; rhat_ik2 = dkz*rik_inv
                    costh = (dx*dkx+dy*dky+dz*dkz) / (rij*rik)
                    costh = clamp(costh, -1.0, 1.0)
                    gv = G_C_d(costh); dgv = dG_C_d(costh)
                    coeff = -vh * dbp

                    # Force on k (x)
                    fk1x = coeff * dfc_ik * gv * rhat_ik0
                    dcx = (rhat0 - costh * rhat_ik0) * rik_inv
                    fkx = fk1x + coeff * fc_ik * dgv * dcx
                    @atomic F[3k-2] += fkx; fi0 -= fkx

                    # Force on k (y)
                    fk1y = coeff * dfc_ik * gv * rhat_ik1
                    dcy = (rhat1 - costh * rhat_ik1) * rik_inv
                    fky = fk1y + coeff * fc_ik * dgv * dcy
                    @atomic F[3k-1] += fky; fi1 -= fky

                    # Force on k (z)
                    fk1z = coeff * dfc_ik * gv * rhat_ik2
                    dcz = (rhat2 - costh * rhat_ik2) * rik_inv
                    fkz = fk1z + coeff * fc_ik * dgv * dcz
                    @atomic F[3k]   += fkz; fi2 -= fkz
                end
            end

            # 3-body: db_ji/dr_l
            if abs(Gs_ji) > 1e-20 && va > 1e-20
                dbp = -BO_DELTA * (1.0+Gs_ji)^(-BO_DELTA-1.0)
                vh = 0.5 * fcut * va
                for ln in 1:nnj
                    l = nll[(j-1)*MAX_REBO_NEIGH+ln]; l == i && continue
                    dlx = pos[3l-2]-pos[3j-2]; dly = pos[3l-1]-pos[3j-1]; dlz = pos[3l]-pos[3j]
                    sl0 = hi[1]*dlx+hi[2]*dly+hi[3]*dlz; sl1 = hi[4]*dlx+hi[5]*dly+hi[6]*dlz; sl2 = hi[7]*dlx+hi[8]*dly+hi[9]*dlz
                    sl0 -= round(sl0); sl1 -= round(sl1); sl2 -= round(sl2)
                    dlx = h[1]*sl0+h[2]*sl1+h[3]*sl2; dly = h[4]*sl0+h[5]*sl1+h[6]*sl2; dlz = h[7]*sl0+h[8]*sl1+h[9]*sl2
                    rjl = sqrt(dlx*dlx+dly*dly+dlz*dlz)
                    rjl > Dmax_CC && continue
                    fc_jl = fc_d(rjl, Dmin_CC, Dmax_CC); fc_jl < 1e-15 && continue
                    dfc_jl = dfc_d(rjl, Dmin_CC, Dmax_CC)
                    rjl_inv = 1.0 / rjl
                    rhat_jl0 = dlx*rjl_inv; rhat_jl1 = dly*rjl_inv; rhat_jl2 = dlz*rjl_inv
                    costh = (-dx*dlx-dy*dly-dz*dlz) / (rij*rjl)
                    costh = clamp(costh, -1.0, 1.0)
                    gv = G_C_d(costh); dgv = dG_C_d(costh)
                    coeff = -vh * dbp
                    rji0 = -rhat0; rji1 = -rhat1; rji2 = -rhat2

                    # Force on l (x)
                    fl1x = coeff * dfc_jl * gv * rhat_jl0
                    dcx = (rji0 - costh * rhat_jl0) * rjl_inv
                    flx = fl1x + coeff * fc_jl * dgv * dcx
                    @atomic F[3l-2] += flx
                    @atomic F[3j-2] -= flx

                    # Force on l (y)
                    fl1y = coeff * dfc_jl * gv * rhat_jl1
                    dcy = (rji1 - costh * rhat_jl1) * rjl_inv
                    fly = fl1y + coeff * fc_jl * dgv * dcy
                    @atomic F[3l-1] += fly
                    @atomic F[3j-1] -= fly

                    # Force on l (z)
                    fl1z = coeff * dfc_jl * gv * rhat_jl2
                    dcz = (rji2 - costh * rhat_jl2) * rjl_inv
                    flz = fl1z + coeff * fc_jl * dgv * dcz
                    @atomic F[3l]   += flz
                    @atomic F[3j]   -= flz
                end
            end
        end

        # Write accumulated F[i] (atomic for safety with 3-body contributions from other threads)
        @atomic F[3i-2] += fi0
        @atomic F[3i-1] += fi1
        @atomic F[3i]   += fi2
        Epbuf[i] = my_Ep
        idx9 = (i-1)*9
        Wbuf[idx9+1]=w00;Wbuf[idx9+2]=w01;Wbuf[idx9+3]=w02
        Wbuf[idx9+4]=w10;Wbuf[idx9+5]=w11;Wbuf[idx9+6]=w12
        Wbuf[idx9+7]=w20;Wbuf[idx9+8]=w21;Wbuf[idx9+9]=w22
    end
    return nothing
end

# LJ intermolecular force kernel (half-list, @atomic for F[j])
function lj_kernel!(i, F, Epbuf, Wbuf, pos, h, hi, nlc, nll)
    @inbounds begin
        fi0 = 0.0; fi1 = 0.0; fi2 = 0.0; my_Ep = 0.0
        w00=0.0;w01=0.0;w02=0.0;w10=0.0;w11=0.0;w12=0.0;w20=0.0;w21=0.0;w22=0.0
        nni = nlc[i]
        for jn in 1:nni
            j = nll[(i-1)*MAX_LJ_NEIGH+jn]
            dx = pos[3j-2]-pos[3i-2]; dy = pos[3j-1]-pos[3i-1]; dz = pos[3j]-pos[3i]
            s0 = hi[1]*dx+hi[2]*dy+hi[3]*dz; s1 = hi[4]*dx+hi[5]*dy+hi[6]*dz; s2 = hi[7]*dx+hi[8]*dy+hi[9]*dz
            s0 -= round(s0); s1 -= round(s1); s2 -= round(s2)
            dx = h[1]*s0+h[2]*s1+h[3]*s2; dy = h[4]*s0+h[5]*s1+h[6]*s2; dz = h[7]*s0+h[8]*s1+h[9]*s2
            r2 = dx*dx+dy*dy+dz*dz
            r2 > LJ_RCUT2_a && continue
            if r2 < 0.25; r2 = 0.25; end
            ri2 = 1.0/r2; sr2 = sig2_LJ_a*ri2; sr6 = sr2*sr2*sr2; sr12 = sr6*sr6
            fm = 24.0*eps_LJ_a*(2.0*sr12-sr6)*ri2
            my_Ep += 4.0*eps_LJ_a*(sr12-sr6) - LJ_VSHFT_a
            fi0 -= fm*dx; fi1 -= fm*dy; fi2 -= fm*dz
            @atomic F[3j-2] += fm*dx
            @atomic F[3j-1] += fm*dy
            @atomic F[3j]   += fm*dz
            w00 += dx*fm*dx; w01 += dx*fm*dy; w02 += dx*fm*dz
            w10 += dy*fm*dx; w11 += dy*fm*dy; w12 += dy*fm*dz
            w20 += dz*fm*dx; w21 += dz*fm*dy; w22 += dz*fm*dz
        end
        @atomic F[3i-2] += fi0
        @atomic F[3i-1] += fi1
        @atomic F[3i]   += fi2
        Epbuf[i] = my_Ep
        idx9 = (i-1)*9
        Wbuf[idx9+1]=w00;Wbuf[idx9+2]=w01;Wbuf[idx9+3]=w02
        Wbuf[idx9+4]=w10;Wbuf[idx9+5]=w11;Wbuf[idx9+6]=w12
        Wbuf[idx9+7]=w20;Wbuf[idx9+8]=w21;Wbuf[idx9+9]=w22
    end
    return nothing
end

# KE kernel (all-atom, mass per atom)
function ke_kernel(i, vel, mass)
    @inbounds mass[i] * (vel[3i-2]^2 + vel[3i-1]^2 + vel[3i]^2)
end

# Velocity pre-update (first half-step)
function vel_pre_kernel!(i, vel, F, mass, sc_v, hdt)
    @inbounds begin
        mi = CONV / mass[i]
        vel[3i-2] = vel[3i-2]*sc_v + hdt*F[3i-2]*mi
        vel[3i-1] = vel[3i-1]*sc_v + hdt*F[3i-1]*mi
        vel[3i]   = vel[3i]*sc_v   + hdt*F[3i]*mi
    end; return nothing
end

# Position update (fractional coordinates + PBC)
function pos_update_kernel!(i, pos, vel, hi, dt_val)
    @inbounds begin
        px=pos[3i-2];py=pos[3i-1];pz=pos[3i];vx=vel[3i-2];vy=vel[3i-1];vz=vel[3i]
        sx=hi[1]*px+hi[2]*py+hi[3]*pz;sy=hi[4]*px+hi[5]*py+hi[6]*pz;sz=hi[7]*px+hi[8]*py+hi[9]*pz
        vsx=hi[1]*vx+hi[2]*vy+hi[3]*vz;vsy=hi[4]*vx+hi[5]*vy+hi[6]*vz;vsz=hi[7]*vx+hi[8]*vy+hi[9]*vz
        sx+=dt_val*vsx;sy+=dt_val*vsy;sz+=dt_val*vsz
        sx-=floor(sx);sy-=floor(sy);sz-=floor(sz)
        pos[3i-2]=sx;pos[3i-1]=sy;pos[3i]=sz
    end; return nothing
end

# Fractional to Cartesian
function frac2cart_kernel!(i, pos, h)
    @inbounds begin
        sx=pos[3i-2];sy=pos[3i-1];sz=pos[3i]
        pos[3i-2]=h[1]*sx+h[2]*sy+h[3]*sz
        pos[3i-1]=h[4]*sx+h[5]*sy+h[6]*sz
        pos[3i]=h[7]*sx+h[8]*sy+h[9]*sz
    end; return nothing
end

# Velocity post-update (second half-step)
function vel_post_kernel!(i, vel, F, mass, sc_v2, hdt)
    @inbounds begin
        mi = CONV / mass[i]
        vel[3i-2] = (vel[3i-2] + hdt*F[3i-2]*mi) * sc_v2
        vel[3i-1] = (vel[3i-1] + hdt*F[3i-1]*mi) * sc_v2
        vel[3i]   = (vel[3i]   + hdt*F[3i]*mi)   * sc_v2
    end; return nothing
end

# Velocity rescaling
function vel_scale_kernel!(i, vel, scale)
    @inbounds begin
        vel[3i-2] *= scale; vel[3i-1] *= scale; vel[3i] *= scale
    end; return nothing
end

# ═══════════ High-level simulation functions ═══════════

function compute_forces_airebo!(F, vir9, Ep_rebo_ref, Ep_lj_ref,
        Epbuf_r, Wbuf_r, Epbuf_l, Wbuf_l,
        pos, h, hi, nlc_r, nll_r, nlc_l, nll_l, Na)
    JACC.parallel_for(Na, zero_kernel!, F)
    JACC.parallel_for(Na, rebo_kernel!, F, Epbuf_r, Wbuf_r, pos, h, hi, nlc_r, nll_r)
    JACC.parallel_for(Na, lj_kernel!, F, Epbuf_l, Wbuf_l, pos, h, hi, nlc_l, nll_l)
    Epbuf_r_h = Array(Epbuf_r); Epbuf_l_h = Array(Epbuf_l)
    Ep_rebo_ref[] = sum(Epbuf_r_h)
    Ep_lj_ref[] = sum(Epbuf_l_h)
    Wbuf_r_h = Array(Wbuf_r); Wbuf_l_h = Array(Wbuf_l)
    for k in 1:9
        s = 0.0
        for i in 1:Na; s += Wbuf_r_h[(i-1)*9+k] + Wbuf_l_h[(i-1)*9+k]; end
        vir9[k] = s
    end
    return nothing
end

function ke_total(vel, mass_d, Na)
    0.5 * JACC.parallel_reduce(Na, ke_kernel, vel, mass_d) / CONV
end

function step_npt_airebo!(pos, vel, F, vir9, h, hi, mass_d, Na, dt, npt,
        Epbuf_r, Wbuf_r, Epbuf_l, Wbuf_l,
        nlc_r, nll_r, nlc_l, nll_l)
    hdt = 0.5 * dt
    V = abs(mat_det9(h))
    KE = ke_total(vel, mass_d, Na)
    npt.xi += hdt * (2*KE - npt.Nf*kB*npt.Tt) / npt.Q
    npt.xi = clamp(npt.xi, -0.05, 0.05)
    dP = inst_P(vir9, KE, V) - npt.Pe
    for a in 0:2; npt.Vg[a*4+1] += hdt*V*dP/(npt.W*eV2GPa); npt.Vg[a*4+1] = clamp(npt.Vg[a*4+1], -0.005, 0.005); end
    h_h = Array(h); hi_h = Array(hi)
    eps_tr = npt.Vg[1]*hi_h[1] + npt.Vg[5]*hi_h[5] + npt.Vg[9]*hi_h[9]
    sc_nh = exp(-hdt * npt.xi); sc_v = sc_nh * exp(-hdt * eps_tr / 3.0)
    JACC.parallel_for(Na, vel_pre_kernel!, vel, F, mass_d, sc_v, hdt)
    JACC.parallel_for(Na, pos_update_kernel!, pos, vel, hi, dt)
    for a in 0:2, b in 0:2; h_h[a*3+b+1] += dt * npt.Vg[a*3+b+1]; end
    mat_inv9!(hi_h, h_h)
    copyto!(h, JACC.array(h_h)); copyto!(hi, JACC.array(hi_h))
    JACC.parallel_for(Na, frac2cart_kernel!, pos, h)
    Ep_rebo_ref = Ref(0.0); Ep_lj_ref = Ref(0.0)
    compute_forces_airebo!(F, vir9, Ep_rebo_ref, Ep_lj_ref,
        Epbuf_r, Wbuf_r, Epbuf_l, Wbuf_l,
        pos, h, hi, nlc_r, nll_r, nlc_l, nll_l, Na)
    eps_tr2 = npt.Vg[1]*hi_h[1] + npt.Vg[5]*hi_h[5] + npt.Vg[9]*hi_h[9]
    sc_v2 = sc_nh * exp(-hdt * eps_tr2 / 3.0)
    JACC.parallel_for(Na, vel_post_kernel!, vel, F, mass_d, sc_v2, hdt)
    KE = ke_total(vel, mass_d, Na)
    npt.xi += hdt * (2*KE - npt.Nf*kB*npt.Tt) / npt.Q
    npt.xi = clamp(npt.xi, -0.05, 0.05)
    V2 = abs(mat_det9(h_h))
    dP = inst_P(vir9, KE, V2) - npt.Pe
    for a in 0:2; npt.Vg[a*4+1] += hdt*V2*dP/(npt.W*eV2GPa); npt.Vg[a*4+1] = clamp(npt.Vg[a*4+1], -0.005, 0.005); end
    return (Ep_rebo_ref[], Ep_lj_ref[], KE)
end

# ═══════════ Restart I/O ═══════════
function write_restart_airebo(fname, istep, opts, st, nc, T, Pe, nsteps, dt, seed, fspec, init_scale,
                              h, npt, pos, vel, mol_id, mass, Na, Nmol, natom_mol)
    open(fname, "w") do f
        print(f, "# RESTART fuller_airebo_npt_md_julia\n# OPTIONS:")
        for (k,v) in opts; print(f, " --$k=$v"); end
        @printf(f, "\nSTEP %d\nNSTEPS %d\nDT %.15e\nTEMP %.15e\nPRES %.15e\n", istep, nsteps, dt, T, Pe)
        @printf(f, "CRYSTAL %s\nNC %d\nFULLERENE %s\nINIT_SCALE %.15e\nSEED %d\n", st, nc, fspec, init_scale, seed)
        @printf(f, "NMOL %d\nNATOM_MOL %d\nNATOM %d\n", Nmol, natom_mol, Na)
        print(f, "H"); for i in 1:9; @printf(f, " %.15e", h[i]); end; print(f, "\n")
        @printf(f, "NPT %.15e %.15e %.15e %.15e %.15e %d\n", npt.xi, npt.Q, npt.W, npt.Pe, npt.Tt, npt.Nf)
        print(f, "VG"); for i in 1:9; @printf(f, " %.15e", npt.Vg[i]); end; print(f, "\n")
        for i in 1:Na
            @printf(f, "ATOM %d %d %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
                i, mol_id[i], mass[i], pos[3i-2], pos[3i-1], pos[3i], vel[3i-2], vel[3i-1], vel[3i])
        end
        print(f, "END\n")
    end
end

function read_restart_airebo(fname)
    h = zeros(9); Vg = zeros(9)
    xi=0.0; Q=0.0; W=0.0; Pe_r=0.0; Tt_r=0.0; Nf=0; istep=0; Na_r=0
    pos_v=Float64[]; vel_v=Float64[]; mol_id_v=Int[]; mass_v=Float64[]
    ok = false
    for line in readlines(fname)
        line = strip(line)
        if isempty(line) || line[1]=='#'; continue; end
        parts = split(line)
        tag = parts[1]
        if tag=="STEP"; istep=parse(Int,parts[2])
        elseif tag=="NATOM"; Na_r=parse(Int,parts[2])
        elseif tag=="H"; for i in 1:9; h[i]=parse(Float64,parts[i+1]); end
        elseif tag=="NPT"
            xi=parse(Float64,parts[2]); Q=parse(Float64,parts[3]); W=parse(Float64,parts[4])
            Pe_r=parse(Float64,parts[5]); Tt_r=parse(Float64,parts[6]); Nf=parse(Int,parts[7])
        elseif tag=="VG"; for i in 1:9; Vg[i]=parse(Float64,parts[i+1]); end
        elseif tag=="ATOM"
            mid = parse(Int, parts[3]); m = parse(Float64, parts[4])
            px=parse(Float64,parts[5]);py=parse(Float64,parts[6]);pz=parse(Float64,parts[7])
            vx=parse(Float64,parts[8]);vy=parse(Float64,parts[9]);vz=parse(Float64,parts[10])
            append!(pos_v, [px,py,pz]); append!(vel_v, [vx,vy,vz])
            push!(mol_id_v, mid); push!(mass_v, m)
        elseif tag=="END"; break; end
    end
    Na_r = div(length(pos_v), 3); ok = Na_r > 0
    npt = NPTState(xi, Q, Vg, W, Pe_r, Tt_r, Nf)
    if ok; @printf("  Restart loaded: %s (step %d, %d atoms)\n", fname, istep, Na_r); end
    return (istep=istep, Na=Na_r, h=h, npt=npt, pos=pos_v, vel=vel_v, mol_id=mol_id_v, mass=mass_v, ok=ok)
end

# ═══════════ MAIN ═══════════
function main()
    opts = parse_args(ARGS)
    if haskey(opts, "help")
        println("""fuller_airebo_npt_md.jl — AIREBO fullerene NPT-MD (JACC.jl)

Options:
  --help                  Show this help
  --fullerene=<name>      Fullerene species (default: C60)
  --crystal=<fcc|hcp|bcc> Crystal structure (default: fcc)
  --cell=<nc>             Unit cell repeats (default: 3)
  --temp=<K>              Target temperature [K] (default: 298.0)
  --pres=<GPa>            Target pressure [GPa] (default: 0.0)
  --step=<N>              Production steps (default: 10000)
  --dt=<fs>               Time step [fs] (default: 0.5)
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
  --libdir=<path>         Fullerene library dir (default: FullereneLib)""")
        return 0
    end

    crystal = get_opt(opts, "crystal", "fcc")
    st = uppercase(crystal)
    nc = parse(Int, get_opt(opts, "cell", "3"))
    T = parse(Float64, get_opt(opts, "temp", "298.0"))
    Pe = parse(Float64, get_opt(opts, "pres", "0.0"))
    nsteps = parse(Int, get_opt(opts, "step", "10000"))
    dt = parse(Float64, get_opt(opts, "dt", "0.5"))
    seed = parse(Int, get_opt(opts, "seed", "42"))
    init_scale = parse(Float64, get_opt(opts, "init_scale", "1.0"))
    fspec = get_opt(opts, "fullerene", "C60")
    libdir = get_opt(opts, "libdir", "FullereneLib")
    coldstart = parse(Int, get_opt(opts, "coldstart", "0"))
    warmup = parse(Int, get_opt(opts, "warmup", "0"))
    avg_from = parse(Int, get_opt(opts, "from", "0"))
    avg_to = parse(Int, get_opt(opts, "to", "0"))
    nrec_o = parse(Int, get_opt(opts, "ovito", "0"))
    nrec_rst = parse(Int, get_opt(opts, "restart", "0"))
    resfile = get_opt(opts, "resfile", "")
    mon_interval = parse(Int, get_opt(opts, "mon", "0"))
    warmup_mon_mode = get_opt(opts, "warmup_mon", "norm")
    T_cold = 4.0

    if avg_to <= 0; avg_to = nsteps; end
    if avg_from <= 0; avg_from = max(1, nsteps - div(nsteps, 4)); end
    total_steps = coldstart + warmup + nsteps
    gavg_from = coldstart + warmup + avg_from
    gavg_to = coldstart + warmup + avg_to

    sfx = build_suffix(opts)
    mode_tag = "julia"
    ovito_file = haskey(opts, "ofile") ? get_opt(opts, "ofile", "") :
                 unique_file("ovito_traj_airebo_" * mode_tag * sfx, ".xyz")

    fpath, label = resolve_fullerene(fspec, libdir)
    mol = load_cc1(fpath)
    natom = mol.natom
    a0 = default_a0(mol.Dmol > 0 ? mol.Dmol : 2*mol.Rmol, st, init_scale)

    Nmol_max = st=="FCC" ? 4*nc^3 : 2*nc^3
    Na_max = Nmol_max * natom

    # Build molecular centers
    mol_centers = zeros(Nmol_max*3)
    h = zeros(9); hi = zeros(9)
    if st=="FCC"; Nmol = make_fcc!(mol_centers, h, a0, nc)
    elseif st=="HCP"; Nmol = make_hcp!(mol_centers, h, a0, nc)
    else; Nmol = make_bcc!(mol_centers, h, a0, nc); end
    Na = Nmol * natom

    # Build all-atom positions from molecular centers + body coordinates
    pos_h = zeros(Na*3); vel_h = zeros(Na*3)
    mass_h = zeros(Na); mol_id_h = zeros(Int, Na)
    for m in 1:Nmol, a in 1:natom
        idx = (m-1)*natom + a
        pos_h[3idx-2] = mol_centers[3m-2] + mol.coords[3a-2]
        pos_h[3idx-1] = mol_centers[3m-1] + mol.coords[3a-1]
        pos_h[3idx]   = mol_centers[3m]   + mol.coords[3a]
        mass_h[idx] = mC
        mol_id_h[idx] = m
    end

    @printf("========================================================================\n")
    @printf("  Fullerene Crystal NPT-MD — AIREBO (JACC.jl)\n")
    @printf("========================================================================\n")
    @printf("  Fullerene       : %s (%d atoms/mol)\n", label, natom)
    @printf("  Crystal         : %s %dx%dx%d  Nmol=%d  Natom=%d\n", st, nc, nc, nc, Nmol, Na)
    @printf("  a0=%.3f A  T=%.1f K  P=%.4f GPa  dt=%.3f fs\n", a0, T, Pe, dt)
    if coldstart>0; @printf("  Coldstart       : %d steps at %.1f K\n", coldstart, T_cold); end
    if warmup>0; @printf("  Warmup          : %d steps (%.1fK->%.1fK)\n", warmup, T_cold, T); end
    @printf("  Production      : %d steps  avg=%d-%d  Total=%d\n", nsteps, avg_from, avg_to, total_steps)
    @printf("========================================================================\n\n")

    mat_inv9!(hi, h)
    T_init = (coldstart>0 || warmup>0) ? T_cold : T
    rng = MersenneTwister(seed)
    for i in 1:Na
        sv = sqrt(kB*T_init*CONV/mass_h[i])
        vel_h[3i-2] = sv*randn(rng); vel_h[3i-1] = sv*randn(rng); vel_h[3i] = sv*randn(rng)
    end
    vcm = zeros(3)
    for i in 1:Na, a in 1:3; vcm[a] += vel_h[3(i-1)+a]; end; vcm ./= Na
    for i in 1:Na, a in 1:3; vel_h[3(i-1)+a] -= vcm[a]; end

    # NPT: all-atom DOF = 3*Na - 3
    Nf = 3*Na - 3
    npt_xi = 0.0; npt_Q = max(Nf*kB*T*100.0^2, 1e-20)
    npt_Vg = zeros(9); npt_W = max((Nf+9)*kB*T*1000.0^2, 1e-20)
    npt = NPTState(npt_xi, npt_Q, npt_Vg, npt_W, Pe, T_init, Nf)

    start_step = 0

    # Load restart if specified
    if !isempty(resfile)
        rd = read_restart_airebo(resfile)
        if rd.ok
            start_step = rd.istep; h .= rd.h; mat_inv9!(hi, h); npt = rd.npt
            nn = min(Na, rd.Na)
            for i in 1:nn
                pos_h[3i-2]=rd.pos[3i-2]; pos_h[3i-1]=rd.pos[3i-1]; pos_h[3i]=rd.pos[3i]
                vel_h[3i-2]=rd.vel[3i-2]; vel_h[3i-1]=rd.vel[3i-1]; vel_h[3i]=rd.vel[3i]
                mol_id_h[i]=rd.mol_id[i]; mass_h[i]=rd.mass[i]
            end
            @printf("  Restarting from global step %d\n", start_step)
        end
    end

    # Transfer to device
    pos = JACC.array(pos_h)
    vel = JACC.array(vel_h)
    F = JACC.zeros(Float64, Na*3)
    mass_d = JACC.array(mass_h)
    h_d = JACC.array(h); hi_d = JACC.array(hi)
    Epbuf_r = JACC.zeros(Float64, Na); Wbuf_r = JACC.zeros(Float64, Na*9)
    Epbuf_l = JACC.zeros(Float64, Na); Wbuf_l = JACC.zeros(Float64, Na*9)
    vir9 = zeros(9)

    # Build neighbor lists (host)
    nlc_r_h = zeros(Int32, Na); nll_r_h = zeros(Int32, Na*MAX_REBO_NEIGH)
    nlc_l_h = zeros(Int32, Na); nll_l_h = zeros(Int32, Na*MAX_LJ_NEIGH)
    build_nlist_rebo!(nlc_r_h, nll_r_h, pos_h, h, hi, Na, mol_id_h)
    build_nlist_lj!(nlc_l_h, nll_l_h, pos_h, h, hi, Na, mol_id_h)
    nlc_r = JACC.array(nlc_r_h); nll_r = JACC.array(nll_r_h)
    nlc_l = JACC.array(nlc_l_h); nll_l = JACC.array(nll_l_h)

    # Initial PBC + forces
    JACC.parallel_for(Na, pbc_kernel!, pos, h_d, hi_d)
    Ep_rebo_ref = Ref(0.0); Ep_lj_ref = Ref(0.0)
    compute_forces_airebo!(F, vir9, Ep_rebo_ref, Ep_lj_ref,
        Epbuf_r, Wbuf_r, Epbuf_l, Wbuf_l,
        pos, h_d, hi_d, nlc_r, nll_r, nlc_l, nll_l, Na)

    prn = mon_interval>0 ? mon_interval : max(1, div(total_steps, 50))
    prn_pre = prn
    if coldstart+warmup>0
        div_val = warmup_mon_mode=="freq" ? 10 : warmup_mon_mode=="some" ? 1000 : 100
        prn_pre = max(1, div(coldstart+warmup, div_val))
    end
    nlup = 20
    sT=0.0; sP=0.0; sa=0.0; sR=0.0; sL=0.0; sE=0.0; nav=0
    t0 = time()

    io_o = nrec_o>0 ? open(ovito_file, "w") : nothing

    @printf("  %8s %5s %7s %9s %8s %11s %11s %11s %7s\n",
        "step","phase","T[K]","P[GPa]","a[A]","E_REBO","E_LJ","E_total","t[s]")

    stop_requested = false
    rst_base = nrec_o>0 ? ovito_file : ("restart_airebo_" * mode_tag)

    for gstep in (start_step+1):total_steps
        phase = gstep<=coldstart ? "COLD" : gstep<=coldstart+warmup ? "WARM" : "PROD"
        cur_prn = gstep<=coldstart+warmup ? prn_pre : prn

        if gstep<=coldstart; npt.Tt=T_cold
        elseif gstep<=coldstart+warmup
            npt.Tt=T_cold+(T-T_cold)*Float64(gstep-coldstart)/Float64(warmup)
        else; npt.Tt=T; end

        if coldstart>0 && gstep==coldstart+1; npt.xi=0.0; fill!(npt.Vg,0.0); end
        if gstep<=coldstart; fill!(npt.Vg,0.0); end

        # Neighbor list rebuild
        if gstep%nlup==0
            pos_h .= Array(pos); mat_inv9!(hi,h)
            build_nlist_rebo!(nlc_r_h, nll_r_h, pos_h, h, hi, Na, mol_id_h)
            build_nlist_lj!(nlc_l_h, nll_l_h, pos_h, h, hi, Na, mol_id_h)
            nlc_r = JACC.array(nlc_r_h); nll_r = JACC.array(nll_r_h)
            nlc_l = JACC.array(nlc_l_h); nll_l = JACC.array(nll_l_h)
            h_d = JACC.array(h); hi_d = JACC.array(hi)
        end

        Ep_rebo, Ep_lj, KE = step_npt_airebo!(pos, vel, F, vir9, h_d, hi_d, mass_d, Na, dt, npt,
            Epbuf_r, Wbuf_r, Epbuf_l, Wbuf_l,
            nlc_r, nll_r, nlc_l, nll_l)
        Ep = Ep_rebo + Ep_lj

        # Sync h back from device
        h .= Array(h_d); hi .= Array(hi_d)

        V = abs(mat_det9(h)); Tn = inst_T(KE, npt.Nf); Pn = inst_P(vir9, KE, V)

        # Cold/warm velocity rescaling
        if (gstep<=coldstart || gstep<=coldstart+warmup) && Tn>0.1
            tgt = gstep<=coldstart ? T_cold : npt.Tt
            scale = sqrt(max(tgt, 0.1) / Tn)
            JACC.parallel_for(Na, vel_scale_kernel!, vel, scale)
            KE = ke_total(vel, mass_d, Na); Tn = inst_T(KE, npt.Nf)
            npt.xi = 0.0; if gstep<=coldstart; fill!(npt.Vg, 0.0); end
        end

        an = h[1] / nc
        if gstep>=gavg_from && gstep<=gavg_to
            sT+=Tn; sP+=Pn; sa+=an; sR+=Ep_rebo/Nmol; sL+=Ep_lj/Nmol; sE+=Ep/Nmol; nav+=1
        end

        # OVITO output
        if io_o!==nothing && gstep%nrec_o==0
            pos_h_o = Array(pos); vel_h_o = Array(vel)
            write_ovito_allatom(io_o, gstep, dt, pos_h_o, vel_h_o, h, Na, mol_id_h)
            flush(io_o)
        end

        # Restart save
        if nrec_rst>0 && (gstep%nrec_rst==0 || gstep==total_steps)
            rfn = restart_filename(rst_base, gstep, total_steps)
            pos_h_r = Array(pos); vel_h_r = Array(vel)
            write_restart_airebo(rfn, gstep, opts, st, nc, T, Pe, nsteps, dt, seed, fspec, init_scale,
                h, npt, pos_h_r, vel_h_r, mol_id_h, mass_h, Na, Nmol, natom)
            if stop_requested
                @printf("\n  *** Stopped at restart checkpoint (step %d) ***\n", gstep); break
            end
        end

        # Monitor + stop control
        if gstep%cur_prn==0 || gstep==total_steps
            if dir_exists("abort.md")
                @printf("\n  *** abort.md detected at step %d ***\n", gstep)
                if nrec_rst>0
                    rfn = restart_filename(rst_base, gstep, total_steps)
                    pos_h_r = Array(pos); vel_h_r = Array(vel)
                    write_restart_airebo(rfn, gstep, opts, st, nc, T, Pe, nsteps, dt, seed, fspec, init_scale,
                        h, npt, pos_h_r, vel_h_r, mol_id_h, mass_h, Na, Nmol, natom)
                end
                break
            end
            if !stop_requested && dir_exists("stop.md")
                stop_requested = true
                @printf("\n  *** stop.md detected at step %d — will stop at next checkpoint ***\n", gstep)
                if nrec_rst == 0; break; end
            end
            el = time() - t0
            @printf("  %8d %5s %7.1f %9.3f %8.3f %11.4f %11.4f %11.4f %7.0f\n",
                gstep, phase, Tn, Pn, an, Ep_rebo/Nmol, Ep_lj/Nmol, Ep/Nmol, el)
        end
    end

    if io_o!==nothing; close(io_o); end
    if nav>0
        @printf("\n========================================================================\n")
        @printf("  Averages (%d): T=%.2f P=%.4f a=%.4f REBO=%.4f LJ=%.4f Total=%.4f\n",
            nav, sT/nav, sP/nav, sa/nav, sR/nav, sL/nav, sE/nav)
        @printf("========================================================================\n")
    end
    @printf("  Done (%.1f sec)\n", time()-t0)
    return 0
end

main()
