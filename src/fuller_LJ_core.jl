#=============================================================================
  fuller_LJ_core.jl — C60 Fullerene Crystal NPT-MD (Core LJ rigid-body)
  Portable CPU/GPU version using JACC.jl

  Usage:
    julia --project=.. fuller_LJ_core.jl [nc]
    julia --project=.. fuller_LJ_core.jl --cell=5

  Parameters fixed in source: T=300K, P=0GPa, dt=1fs, 1000 steps.
  No restart, no OVITO output.

  Unit system: A, amu, eV, fs, K, GPa
=============================================================================#

using JACC
using Printf, Random
JACC.@init_backend

include("FullerMD.jl")
using .FullerMD

# ═══════════ JACC Kernel Functions ═══════════
# All kernel functions: first arg is loop index i (1-based)

# Compute lab-frame atom coordinates from quaternion rotation
function lab_coords_kernel!(i, lab, pos, qv, body, natom)
    @inbounds begin
        w = qv[4i-3]; x = qv[4i-2]; y = qv[4i-1]; z = qv[4i]
        R11 = 1-2(y*y+z*z); R12 = 2(x*y-w*z); R13 = 2(x*z+w*y)
        R21 = 2(x*y+w*z);   R22 = 1-2(x*x+z*z); R23 = 2(y*z-w*x)
        R31 = 2(x*z-w*y);   R32 = 2(y*z+w*x);   R33 = 1-2(x*x+y*y)
        base = (i-1)*natom*3
        for a in 1:natom
            bx = body[3a-2]; by = body[3a-1]; bz = body[3a]
            idx = base + (a-1)*3
            lab[idx+1] = R11*bx + R12*by + R13*bz
            lab[idx+2] = R21*bx + R22*by + R23*bz
            lab[idx+3] = R31*bx + R32*by + R33*bz
        end
    end
    return nothing
end

# Main LJ force kernel — symmetric full list, each i writes only to its own F[i]
function force_kernel!(i, Fv, Tv, Wbuf, Epbuf,
                       pos, lab, nl_count, nl_list,
                       h, hi, natom, max_neigh, rmcut2,
                       sig2, eps4, eps24, vshft, rcut2)
    @inbounds begin
        fi0 = 0.0; fi1 = 0.0; fi2 = 0.0
        ti0 = 0.0; ti1 = 0.0; ti2 = 0.0
        my_Ep = 0.0
        w00=0.0;w01=0.0;w02=0.0;w10=0.0;w11=0.0;w12=0.0;w20=0.0;w21=0.0;w22=0.0

        nni = nl_count[i]
        for k in 1:nni
            j = nl_list[(i-1)*max_neigh + k]
            # Minimum image for molecular pair
            dmx = pos[3j-2] - pos[3i-2]
            dmy = pos[3j-1] - pos[3i-1]
            dmz = pos[3j]   - pos[3i]
            # Inline mimg
            s0 = hi[1]*dmx+hi[2]*dmy+hi[3]*dmz
            s1 = hi[4]*dmx+hi[5]*dmy+hi[6]*dmz
            s2 = hi[7]*dmx+hi[8]*dmy+hi[9]*dmz
            s0 -= round(s0); s1 -= round(s1); s2 -= round(s2)
            dmx = h[1]*s0+h[2]*s1+h[3]*s2
            dmy = h[4]*s0+h[5]*s1+h[6]*s2
            dmz = h[7]*s0+h[8]*s1+h[9]*s2

            if dmx*dmx+dmy*dmy+dmz*dmz > rmcut2; continue; end

            base_i = (i-1)*natom*3
            base_j = (j-1)*natom*3
            for ai in 1:natom
                ia = base_i + (ai-1)*3
                rax = lab[ia+1]; ray = lab[ia+2]; raz = lab[ia+3]
                for bj in 1:natom
                    jb = base_j + (bj-1)*3
                    ddx = dmx + lab[jb+1] - rax
                    ddy = dmy + lab[jb+2] - ray
                    ddz = dmz + lab[jb+3] - raz
                    r2 = ddx*ddx + ddy*ddy + ddz*ddz
                    if r2 < rcut2
                        if r2 < 0.25; r2 = 0.25; end
                        ri2 = 1.0/r2
                        sr2 = sig2*ri2
                        sr6 = sr2*sr2*sr2
                        sr12 = sr6*sr6
                        fm = eps24*(2.0*sr12 - sr6)*ri2
                        fx = fm*ddx; fy = fm*ddy; fz = fm*ddz
                        fi0 -= fx; fi1 -= fy; fi2 -= fz
                        ti0 -= (ray*fz - raz*fy)
                        ti1 -= (raz*fx - rax*fz)
                        ti2 -= (rax*fy - ray*fx)
                        my_Ep += 0.5*(eps4*(sr12-sr6) - vshft)
                        w00+=0.5*ddx*fx;w01+=0.5*ddx*fy;w02+=0.5*ddx*fz
                        w10+=0.5*ddy*fx;w11+=0.5*ddy*fy;w12+=0.5*ddy*fz
                        w20+=0.5*ddz*fx;w21+=0.5*ddz*fy;w22+=0.5*ddz*fz
                    end
                end
            end
        end
        Fv[3i-2] = fi0; Fv[3i-1] = fi1; Fv[3i] = fi2
        Tv[3i-2] = ti0; Tv[3i-1] = ti1; Tv[3i] = ti2
        idx9 = (i-1)*9
        Wbuf[idx9+1]=w00;Wbuf[idx9+2]=w01;Wbuf[idx9+3]=w02
        Wbuf[idx9+4]=w10;Wbuf[idx9+5]=w11;Wbuf[idx9+6]=w12
        Wbuf[idx9+7]=w20;Wbuf[idx9+8]=w21;Wbuf[idx9+9]=w22
        Epbuf[i] = my_Ep
    end
    return nothing
end

# KE translation kernel (returns per-molecule contribution)
function ke_trans_kernel(i, vel)
    @inbounds vel[3i-2]^2 + vel[3i-1]^2 + vel[3i]^2
end

# KE rotation kernel
function ke_rot_kernel(i, omg)
    @inbounds omg[3i-2]^2 + omg[3i-1]^2 + omg[3i]^2
end

# Velocity pre-half update kernel
function vel_pre_kernel!(i, vel, omg, Fv, Tv, sc_v, sc_nh, hdt_cF, hdt_cT)
    @inbounds begin
        vel[3i-2] = vel[3i-2]*sc_v + hdt_cF*Fv[3i-2]
        vel[3i-1] = vel[3i-1]*sc_v + hdt_cF*Fv[3i-1]
        vel[3i]   = vel[3i]*sc_v   + hdt_cF*Fv[3i]
        omg[3i-2] = omg[3i-2]*sc_nh + hdt_cT*Tv[3i-2]
        omg[3i-1] = omg[3i-1]*sc_nh + hdt_cT*Tv[3i-1]
        omg[3i]   = omg[3i]*sc_nh   + hdt_cT*Tv[3i]
    end
    return nothing
end

# Position update kernel (to fractional, integrate, PBC)
function pos_update_kernel!(i, pos, vel, hi, dt_val)
    @inbounds begin
        px=pos[3i-2];py=pos[3i-1];pz=pos[3i]
        vx=vel[3i-2];vy=vel[3i-1];vz=vel[3i]
        sx=hi[1]*px+hi[2]*py+hi[3]*pz
        sy=hi[4]*px+hi[5]*py+hi[6]*pz
        sz=hi[7]*px+hi[8]*py+hi[9]*pz
        vsx=hi[1]*vx+hi[2]*vy+hi[3]*vz
        vsy=hi[4]*vx+hi[5]*vy+hi[6]*vz
        vsz=hi[7]*vx+hi[8]*vy+hi[9]*vz
        sx+=dt_val*vsx; sy+=dt_val*vsy; sz+=dt_val*vsz
        sx-=floor(sx); sy-=floor(sy); sz-=floor(sz)
        pos[3i-2]=sx; pos[3i-1]=sy; pos[3i]=sz
    end
    return nothing
end

# Fractional to Cartesian kernel
function frac2cart_kernel!(i, pos, h)
    @inbounds begin
        sx=pos[3i-2]; sy=pos[3i-1]; sz=pos[3i]
        pos[3i-2] = h[1]*sx+h[2]*sy+h[3]*sz
        pos[3i-1] = h[4]*sx+h[5]*sy+h[6]*sz
        pos[3i]   = h[7]*sx+h[8]*sy+h[9]*sz
    end
    return nothing
end

# Quaternion update kernel
function quat_update_kernel!(i, qv, omg, dt_val)
    @inbounds begin
        wx=omg[3i-2]; wy=omg[3i-1]; wz=omg[3i]
        wm = sqrt(wx*wx+wy*wy+wz*wz)
        th = wm*dt_val*0.5
        if th < 1e-14
            dw=1.0; dx=0.5*dt_val*wx; dy=0.5*dt_val*wy; dz=0.5*dt_val*wz
        else
            s = sin(th)/wm
            dw=cos(th); dx=s*wx; dy=s*wy; dz=s*wz
        end
        aw=qv[4i-3]; ax=qv[4i-2]; ay=qv[4i-1]; az=qv[4i]
        ow = aw*dw-ax*dx-ay*dy-az*dz
        ox = aw*dx+ax*dw+ay*dz-az*dy
        oy = aw*dy-ax*dz+ay*dw+az*dx
        oz = aw*dz+ax*dy-ay*dx+az*dw
        n = sqrt(ow*ow+ox*ox+oy*oy+oz*oz)
        inv_n = 1.0/n
        qv[4i-3]=ow*inv_n; qv[4i-2]=ox*inv_n; qv[4i-1]=oy*inv_n; qv[4i]=oz*inv_n
    end
    return nothing
end

# Velocity post-half update kernel
function vel_post_kernel!(i, vel, omg, Fv, Tv, sc_v2, sc_nh, hdt_cF, hdt_cT)
    @inbounds begin
        vel[3i-2] = (vel[3i-2]+hdt_cF*Fv[3i-2])*sc_v2
        vel[3i-1] = (vel[3i-1]+hdt_cF*Fv[3i-1])*sc_v2
        vel[3i]   = (vel[3i]+hdt_cF*Fv[3i])*sc_v2
        omg[3i-2] = (omg[3i-2]+hdt_cT*Tv[3i-2])*sc_nh
        omg[3i-1] = (omg[3i-1]+hdt_cT*Tv[3i-1])*sc_nh
        omg[3i]   = (omg[3i]+hdt_cT*Tv[3i])*sc_nh
    end
    return nothing
end

# Velocity scale kernel (for cold-start/warmup)
function vel_scale_kernel!(i, vel, omg, scale)
    @inbounds begin
        vel[3i-2]*=scale; vel[3i-1]*=scale; vel[3i]*=scale
        omg[3i-2]*=scale; omg[3i-1]*=scale; omg[3i]*=scale
    end
    return nothing
end

# PBC kernel
function pbc_kernel!(i, pos, h, hi)
    @inbounds begin
        px=pos[3i-2]; py=pos[3i-1]; pz=pos[3i]
        s0=hi[1]*px+hi[2]*py+hi[3]*pz
        s1=hi[4]*px+hi[5]*py+hi[6]*pz
        s2=hi[7]*px+hi[8]*py+hi[9]*pz
        s0-=floor(s0); s1-=floor(s1); s2-=floor(s2)
        pos[3i-2]=h[1]*s0+h[2]*s1+h[3]*s2
        pos[3i-1]=h[4]*s0+h[5]*s1+h[6]*s2
        pos[3i]  =h[7]*s0+h[8]*s1+h[9]*s2
    end
    return nothing
end

# ═══════════ High-level functions ═══════════

function compute_forces!(Fv, Tv, Wm9, Ep_ref, Wbuf, Epbuf,
                         pos, qv, body, h, hi,
                         nl_count, nl_list, lab,
                         N, natom, rmcut2)
    # Step 1: lab coordinates
    JACC.parallel_for(N, lab_coords_kernel!, lab, pos, qv, body, natom)
    # Step 2: forces
    eps4 = 4.0*eps_LJ; eps24 = 24.0*eps_LJ
    JACC.parallel_for(N, force_kernel!, Fv, Tv, Wbuf, Epbuf,
        pos, lab, nl_count, nl_list, h, hi,
        natom, MAX_NEIGH, rmcut2, sig2_LJ, eps4, eps24, VSHFT, RCUT2)
    # Step 3: reduce virial and energy
    Wbuf_h = Array(Wbuf)
    Epbuf_h = Array(Epbuf)
    Ep_ref[] = sum(Epbuf_h)
    for k in 1:9
        s = 0.0
        for i in 1:N
            s += Wbuf_h[(i-1)*9 + k]
        end
        Wm9[k] = s
    end
    return nothing
end

function ke_trans(vel, N, Mmol)
    s = JACC.parallel_reduce(N, ke_trans_kernel, vel)
    return 0.5 * Mmol * s / CONV
end

function ke_rot(omg, N, I0)
    s = JACC.parallel_reduce(N, ke_rot_kernel, omg)
    return 0.5 * I0 * s / CONV
end

function step_npt!(pos, vel, qv, omg, Fv, Tv, Wm9, Wbuf, Epbuf,
                   h, hi, body, I0, Mmol, N, natom, rmcut2,
                   dt, npt, nl_count, nl_list, lab)
    hdt = 0.5*dt
    mat_inv9!(hi, h)

    V = abs(mat_det9(h))
    kt = ke_trans(vel, N, Mmol)
    kr = ke_rot(omg, N, I0)
    KE = kt + kr

    # (A) Thermostat pre-half
    npt.xi += hdt*(2*KE - npt.Nf*kB*npt.Tt)/npt.Q
    npt.xi = clamp(npt.xi, -0.1, 0.1)

    # (B) Barostat pre-half
    Wm9_h = Array(Wm9)
    dP = inst_P(Wm9_h, kt, V) - npt.Pe
    for a in 0:2
        npt.Vg[a*4+1] += hdt*V*dP/(npt.W*eV2GPa)
        npt.Vg[a*4+1] = clamp(npt.Vg[a*4+1], -0.01, 0.01)
    end

    hi_h = Array(hi)
    eps_tr = npt.Vg[1]*hi_h[1] + npt.Vg[5]*hi_h[5] + npt.Vg[9]*hi_h[9]
    sc_nh = exp(-hdt*npt.xi)
    sc_pr = exp(-hdt*eps_tr/3.0)
    sc_v = sc_nh*sc_pr
    hdt_cF = hdt*CONV/Mmol
    hdt_cT = hdt*CONV/I0

    # (C) Velocity pre-half
    JACC.parallel_for(N, vel_pre_kernel!, vel, omg, Fv, Tv, sc_v, sc_nh, hdt_cF, hdt_cT)

    # (D) Position update (fractional coords + PBC)
    JACC.parallel_for(N, pos_update_kernel!, pos, vel, hi, dt)

    # (E) Cell H-matrix update (host)
    h_h = Array(h)
    for a in 0:2, b in 0:2
        h_h[a*3+b+1] += dt*npt.Vg[a*3+b+1]
    end
    copyto!(h, JACC.array(h_h))

    # (F) Fractional→Cartesian
    JACC.parallel_for(N, frac2cart_kernel!, pos, h)

    # (G) Quaternion update
    JACC.parallel_for(N, quat_update_kernel!, qv, omg, dt)

    # (H) Force recalculation
    mat_inv9!(hi_h, h_h)
    copyto!(hi, JACC.array(hi_h))
    Ep_ref = Ref(0.0)
    compute_forces!(Fv, Tv, Wm9, Ep_ref, Wbuf, Epbuf,
                    pos, qv, body, h, hi, nl_count, nl_list, lab,
                    N, natom, rmcut2)

    # (I) Velocity post-half
    hi_h2 = Array(hi)
    eps_tr2 = npt.Vg[1]*hi_h2[1] + npt.Vg[5]*hi_h2[5] + npt.Vg[9]*hi_h2[9]
    sc_v2 = sc_nh*exp(-hdt*eps_tr2/3.0)
    JACC.parallel_for(N, vel_post_kernel!, vel, omg, Fv, Tv, sc_v2, sc_nh, hdt_cF, hdt_cT)

    # (J)(K) Thermostat/Barostat post-half
    kt = ke_trans(vel, N, Mmol)
    kr = ke_rot(omg, N, I0)
    KE = kt + kr
    npt.xi += hdt*(2*KE - npt.Nf*kB*npt.Tt)/npt.Q
    npt.xi = clamp(npt.xi, -0.1, 0.1)
    V2 = abs(mat_det9(Array(h)))
    Wm9_h2 = Array(Wm9)
    dP = inst_P(Wm9_h2, kt, V2) - npt.Pe
    for a in 0:2
        npt.Vg[a*4+1] += hdt*V2*dP/(npt.W*eV2GPa)
        npt.Vg[a*4+1] = clamp(npt.Vg[a*4+1], -0.01, 0.01)
    end

    return (Ep_ref[], KE)
end

# ═══════════ MAIN ═══════════
function main()
    # Parse cell size
    nc = 3
    for a in ARGS
        if startswith(a, "--cell=")
            nc = parse(Int, a[8:end])
        elseif !startswith(a, "-")
            nc = parse(Int, a)
        end
    end
    if nc < 1 || nc > 8
        println("Error: nc must be 1-8 (got $nc)")
        return 1
    end

    # Fixed parameters
    nsteps = 1000
    mon = 100
    nlup = 25
    T = 300.0
    Pe = 0.0
    dt = 1.0
    a0 = 14.17
    avg_from = nsteps - div(nsteps, 4)

    # Generate C60
    c60 = generate_c60()
    natom = c60.natom
    RMCUT = RCUT + 2*c60.Rmol + 1.0
    RMCUT2 = RMCUT*RMCUT

    # Allocate arrays
    N = 4*nc^3
    pos = JACC.zeros(Float64, N*3)
    vel = JACC.zeros(Float64, N*3)
    omg = JACC.zeros(Float64, N*3)
    qv  = JACC.zeros(Float64, N*4)
    Fv  = JACC.zeros(Float64, N*3)
    Tv  = JACC.zeros(Float64, N*3)
    lab = JACC.zeros(Float64, N*natom*3)
    body = JACC.array(copy(c60.coords))
    h   = JACC.zeros(Float64, 9)
    hi  = JACC.zeros(Float64, 9)
    Wm9 = JACC.zeros(Float64, 9)
    Wbuf  = JACC.zeros(Float64, N*9)
    Epbuf = JACC.zeros(Float64, N)
    nl_count = zeros(Int32, N)
    nl_list  = zeros(Int32, N*MAX_NEIGH)

    # Build FCC crystal (on host, then copy)
    pos_h = zeros(N*3)
    h_h = zeros(9)
    N_built = make_fcc!(pos_h, h_h, a0, nc)
    copyto!(pos, JACC.array(pos_h))
    copyto!(h, JACC.array(h_h))
    hi_h = zeros(9)
    mat_inv9!(hi_h, h_h)
    copyto!(hi, JACC.array(hi_h))

    # Banner
    @printf("================================================================\n")
    @printf("  C60 LJ NPT-MD Core (JACC.jl)\n")
    @printf("================================================================\n")
    @printf("  FCC cell        : %dx%dx%d  N=%d molecules\n", nc, nc, nc, N)
    @printf("  Atoms/molecule  : %d\n", natom)
    @printf("  a0=%.2f A  T=%.0f K  P=%.1f GPa  dt=%.1f fs  steps=%d\n", a0, T, Pe, dt, nsteps)
    @printf("  MAX_NEIGH=%d\n", MAX_NEIGH)
    @printf("================================================================\n\n")

    # Initial velocities (Maxwell-Boltzmann)
    rng = MersenneTwister(42)
    sv = sqrt(kB*T*CONV/c60.Mmol)
    sw = sqrt(kB*T*CONV/c60.I0)
    vel_h = zeros(N*3); omg_h = zeros(N*3); qv_h = zeros(N*4)
    for i in 1:N
        for a in 1:3
            vel_h[3(i-1)+a] = sv * randn(rng)
            omg_h[3(i-1)+a] = sw * randn(rng)
        end
        for a in 1:4
            qv_h[4(i-1)+a] = randn(rng)
        end
        n = sqrt(sum(qv_h[4(i-1)+a]^2 for a in 1:4))
        for a in 1:4; qv_h[4(i-1)+a] /= n; end
    end
    # Remove CoM velocity
    vcm = zeros(3)
    for i in 1:N, a in 1:3; vcm[a] += vel_h[3(i-1)+a]; end
    vcm ./= N
    for i in 1:N, a in 1:3; vel_h[3(i-1)+a] -= vcm[a]; end

    copyto!(vel, JACC.array(vel_h))
    copyto!(omg, JACC.array(omg_h))
    copyto!(qv, JACC.array(qv_h))

    npt = make_npt(T, Pe, N)

    # Build initial neighbor list (host)
    pos_h2 = Array(pos)
    nlist_build_sym!(nl_count, nl_list, pos_h2, h_h, hi_h, N, RMCUT, MAX_NEIGH)
    # Convert to JACC arrays for kernel use
    nl_count_d = JACC.array(nl_count)
    nl_list_d  = JACC.array(nl_list)

    # Initial PBC + force calculation
    JACC.parallel_for(N, pbc_kernel!, pos, h, hi)
    Ep_ref = Ref(0.0)
    compute_forces!(Fv, Tv, Wm9, Ep_ref, Wbuf, Epbuf,
                    pos, qv, body, h, hi, nl_count_d, nl_list_d, lab,
                    N, natom, RMCUT2)

    sT=0.0; sP=0.0; sa=0.0; sEp=0.0; nav=0
    t0 = time()
    @printf("%8s %7s %9s %8s %10s %7s\n", "step", "T[K]", "P[GPa]", "a[A]", "Ecoh[eV]", "t[s]")

    # ═══ MD Main Loop ═══
    for g in 1:nsteps
        # Neighbor list rebuild
        if g % nlup == 0
            pos_h_tmp = Array(pos)
            h_h_tmp = Array(h)
            hi_h_tmp = zeros(9)
            mat_inv9!(hi_h_tmp, h_h_tmp)
            nlist_build_sym!(nl_count, nl_list, pos_h_tmp, h_h_tmp, hi_h_tmp, N, RMCUT, MAX_NEIGH)
            nl_count_d = JACC.array(nl_count)
            nl_list_d  = JACC.array(nl_list)
        end

        Ep, KE = step_npt!(pos, vel, qv, omg, Fv, Tv, Wm9, Wbuf, Epbuf,
                            h, hi, body, c60.I0, c60.Mmol,
                            N, natom, RMCUT2, dt, npt,
                            nl_count_d, nl_list_d, lab)

        kt = ke_trans(vel, N, c60.Mmol)
        h_h_tmp = Array(h)
        V = abs(mat_det9(h_h_tmp))
        Wm9_h = Array(Wm9)
        Tn = inst_T(KE, npt.Nf)
        Pn = inst_P(Wm9_h, kt, V)
        Ec = Ep / N
        an = h_h_tmp[1] / nc
        if g >= avg_from; sT+=Tn; sP+=Pn; sa+=an; sEp+=Ec; nav+=1; end

        if g % mon == 0 || g == nsteps
            el = time() - t0
            @printf("%8d %7.1f %9.3f %8.3f %10.5f %7.0f\n", g, Tn, Pn, an, Ec, el)
        end
    end

    if nav > 0
        @printf("Avg(%d): T=%.2f P=%.4f a=%.4f Ecoh=%.5f\n",
                nav, sT/nav, sP/nav, sa/nav, sEp/nav)
    end
    @printf("Done %.1fs\n", time() - t0)
    return 0
end

main()
