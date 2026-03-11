# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Takeshi Nishikawa
#=============================================================================
  fuller_LJ_npt_md.jl — Fullerene Crystal NPT-MD (Full LJ rigid-body)
  Portable CPU/GPU version using JACC.jl

  Usage:
    julia --project=.. fuller_LJ_npt_md.jl [options]

  Options:
    --help                  Show help
    --fullerene=<name>      Fullerene species (default: C60)
    --crystal=<fcc|hcp|bcc> Crystal structure (default: fcc)
    --cell=<nc>             Unit cell repeats (default: 3)
    --temp=<K>              Target temperature [K] (default: 298.0)
    --pres=<GPa>            Target pressure [GPa] (default: 0.0)
    --step=<N>              Production steps (default: 10000)
    --dt=<fs>               Time step [fs] (default: 1.0)
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

# ═══════════ JACC Kernel Functions ═══════════
# (Same kernels as fuller_LJ_core.jl — included here for standalone operation)

function lab_coords_kernel!(i, lab, pos, qv, body, natom)
    @inbounds begin
        w=qv[4i-3]; x=qv[4i-2]; y=qv[4i-1]; z=qv[4i]
        R11=1-2(y*y+z*z); R12=2(x*y-w*z); R13=2(x*z+w*y)
        R21=2(x*y+w*z);   R22=1-2(x*x+z*z); R23=2(y*z-w*x)
        R31=2(x*z-w*y);   R32=2(y*z+w*x);   R33=1-2(x*x+y*y)
        base=(i-1)*natom*3
        for a in 1:natom
            bx=body[3a-2]; by=body[3a-1]; bz=body[3a]
            idx=base+(a-1)*3
            lab[idx+1]=R11*bx+R12*by+R13*bz
            lab[idx+2]=R21*bx+R22*by+R23*bz
            lab[idx+3]=R31*bx+R32*by+R33*bz
        end
    end
    return nothing
end

function force_kernel!(i, Fv, Tv, Wbuf, Epbuf, pos, lab, nl_count, nl_list,
                       h, hi, natom, max_neigh, rmcut2, sig2, eps4, eps24, vshft, rcut2)
    @inbounds begin
        fi0=0.0;fi1=0.0;fi2=0.0; ti0=0.0;ti1=0.0;ti2=0.0; my_Ep=0.0
        w00=0.0;w01=0.0;w02=0.0;w10=0.0;w11=0.0;w12=0.0;w20=0.0;w21=0.0;w22=0.0
        nni=nl_count[i]
        for k in 1:nni
            j=nl_list[(i-1)*max_neigh+k]
            dmx=pos[3j-2]-pos[3i-2]; dmy=pos[3j-1]-pos[3i-1]; dmz=pos[3j]-pos[3i]
            s0=hi[1]*dmx+hi[2]*dmy+hi[3]*dmz; s1=hi[4]*dmx+hi[5]*dmy+hi[6]*dmz
            s2=hi[7]*dmx+hi[8]*dmy+hi[9]*dmz
            s0-=round(s0);s1-=round(s1);s2-=round(s2)
            dmx=h[1]*s0+h[2]*s1+h[3]*s2; dmy=h[4]*s0+h[5]*s1+h[6]*s2; dmz=h[7]*s0+h[8]*s1+h[9]*s2
            if dmx*dmx+dmy*dmy+dmz*dmz>rmcut2; continue; end
            bi=(i-1)*natom*3; bj_base=(j-1)*natom*3
            for ai in 1:natom
                ia=bi+(ai-1)*3; rax=lab[ia+1];ray=lab[ia+2];raz=lab[ia+3]
                for bj in 1:natom
                    jb=bj_base+(bj-1)*3
                    ddx=dmx+lab[jb+1]-rax; ddy=dmy+lab[jb+2]-ray; ddz=dmz+lab[jb+3]-raz
                    r2=ddx*ddx+ddy*ddy+ddz*ddz
                    if r2<rcut2
                        if r2<0.25; r2=0.25; end
                        ri2=1.0/r2;sr2=sig2*ri2;sr6=sr2*sr2*sr2;sr12=sr6*sr6
                        fm=eps24*(2.0*sr12-sr6)*ri2
                        fx=fm*ddx;fy=fm*ddy;fz=fm*ddz
                        fi0-=fx;fi1-=fy;fi2-=fz
                        ti0-=(ray*fz-raz*fy);ti1-=(raz*fx-rax*fz);ti2-=(rax*fy-ray*fx)
                        my_Ep+=0.5*(eps4*(sr12-sr6)-vshft)
                        w00+=0.5*ddx*fx;w01+=0.5*ddx*fy;w02+=0.5*ddx*fz
                        w10+=0.5*ddy*fx;w11+=0.5*ddy*fy;w12+=0.5*ddy*fz
                        w20+=0.5*ddz*fx;w21+=0.5*ddz*fy;w22+=0.5*ddz*fz
                    end
                end
            end
        end
        Fv[3i-2]=fi0;Fv[3i-1]=fi1;Fv[3i]=fi2
        Tv[3i-2]=ti0;Tv[3i-1]=ti1;Tv[3i]=ti2
        idx9=(i-1)*9
        Wbuf[idx9+1]=w00;Wbuf[idx9+2]=w01;Wbuf[idx9+3]=w02
        Wbuf[idx9+4]=w10;Wbuf[idx9+5]=w11;Wbuf[idx9+6]=w12
        Wbuf[idx9+7]=w20;Wbuf[idx9+8]=w21;Wbuf[idx9+9]=w22
        Epbuf[i]=my_Ep
    end
    return nothing
end

function ke_trans_kernel(i, vel); @inbounds vel[3i-2]^2+vel[3i-1]^2+vel[3i]^2; end
function ke_rot_kernel(i, omg); @inbounds omg[3i-2]^2+omg[3i-1]^2+omg[3i]^2; end

function vel_pre_kernel!(i, vel, omg, Fv, Tv, sc_v, sc_nh, hdt_cF, hdt_cT)
    @inbounds begin
        vel[3i-2]=vel[3i-2]*sc_v+hdt_cF*Fv[3i-2]; vel[3i-1]=vel[3i-1]*sc_v+hdt_cF*Fv[3i-1]; vel[3i]=vel[3i]*sc_v+hdt_cF*Fv[3i]
        omg[3i-2]=omg[3i-2]*sc_nh+hdt_cT*Tv[3i-2]; omg[3i-1]=omg[3i-1]*sc_nh+hdt_cT*Tv[3i-1]; omg[3i]=omg[3i]*sc_nh+hdt_cT*Tv[3i]
    end; return nothing; end

function pos_update_kernel!(i, pos, vel, hi, dt_val)
    @inbounds begin
        px=pos[3i-2];py=pos[3i-1];pz=pos[3i];vx=vel[3i-2];vy=vel[3i-1];vz=vel[3i]
        sx=hi[1]*px+hi[2]*py+hi[3]*pz;sy=hi[4]*px+hi[5]*py+hi[6]*pz;sz=hi[7]*px+hi[8]*py+hi[9]*pz
        vsx=hi[1]*vx+hi[2]*vy+hi[3]*vz;vsy=hi[4]*vx+hi[5]*vy+hi[6]*vz;vsz=hi[7]*vx+hi[8]*vy+hi[9]*vz
        sx+=dt_val*vsx;sy+=dt_val*vsy;sz+=dt_val*vsz
        sx-=floor(sx);sy-=floor(sy);sz-=floor(sz)
        pos[3i-2]=sx;pos[3i-1]=sy;pos[3i]=sz
    end; return nothing; end

function frac2cart_kernel!(i, pos, h)
    @inbounds begin
        sx=pos[3i-2];sy=pos[3i-1];sz=pos[3i]
        pos[3i-2]=h[1]*sx+h[2]*sy+h[3]*sz;pos[3i-1]=h[4]*sx+h[5]*sy+h[6]*sz;pos[3i]=h[7]*sx+h[8]*sy+h[9]*sz
    end; return nothing; end

function quat_update_kernel!(i, qv, omg, dt_val)
    @inbounds begin
        wx=omg[3i-2];wy=omg[3i-1];wz=omg[3i]
        wm=sqrt(wx*wx+wy*wy+wz*wz); th=wm*dt_val*0.5
        if th<1e-14; dw=1.0;dx=0.5*dt_val*wx;dy=0.5*dt_val*wy;dz=0.5*dt_val*wz
        else; s=sin(th)/wm;dw=cos(th);dx=s*wx;dy=s*wy;dz=s*wz; end
        aw=qv[4i-3];ax=qv[4i-2];ay=qv[4i-1];az=qv[4i]
        ow=aw*dw-ax*dx-ay*dy-az*dz;ox=aw*dx+ax*dw+ay*dz-az*dy
        oy=aw*dy-ax*dz+ay*dw+az*dx;oz=aw*dz+ax*dy-ay*dx+az*dw
        n=sqrt(ow*ow+ox*ox+oy*oy+oz*oz);inv_n=1.0/n
        qv[4i-3]=ow*inv_n;qv[4i-2]=ox*inv_n;qv[4i-1]=oy*inv_n;qv[4i]=oz*inv_n
    end; return nothing; end

function vel_post_kernel!(i, vel, omg, Fv, Tv, sc_v2, sc_nh, hdt_cF, hdt_cT)
    @inbounds begin
        vel[3i-2]=(vel[3i-2]+hdt_cF*Fv[3i-2])*sc_v2;vel[3i-1]=(vel[3i-1]+hdt_cF*Fv[3i-1])*sc_v2;vel[3i]=(vel[3i]+hdt_cF*Fv[3i])*sc_v2
        omg[3i-2]=(omg[3i-2]+hdt_cT*Tv[3i-2])*sc_nh;omg[3i-1]=(omg[3i-1]+hdt_cT*Tv[3i-1])*sc_nh;omg[3i]=(omg[3i]+hdt_cT*Tv[3i])*sc_nh
    end; return nothing; end

function vel_scale_kernel!(i, vel, omg, scale)
    @inbounds begin
        vel[3i-2]*=scale;vel[3i-1]*=scale;vel[3i]*=scale
        omg[3i-2]*=scale;omg[3i-1]*=scale;omg[3i]*=scale
    end; return nothing; end

function pbc_kernel!(i, pos, h, hi)
    @inbounds begin
        px=pos[3i-2];py=pos[3i-1];pz=pos[3i]
        s0=hi[1]*px+hi[2]*py+hi[3]*pz;s1=hi[4]*px+hi[5]*py+hi[6]*pz;s2=hi[7]*px+hi[8]*py+hi[9]*pz
        s0-=floor(s0);s1-=floor(s1);s2-=floor(s2)
        pos[3i-2]=h[1]*s0+h[2]*s1+h[3]*s2;pos[3i-1]=h[4]*s0+h[5]*s1+h[6]*s2;pos[3i]=h[7]*s0+h[8]*s1+h[9]*s2
    end; return nothing; end

# ═══════════ High-level simulation functions ═══════════

function compute_forces!(Fv, Tv, Wm9, Ep_ref, Wbuf, Epbuf,
                         pos, qv, body, h, hi, nl_count, nl_list, lab, N, natom, rmcut2)
    JACC.parallel_for(N, lab_coords_kernel!, lab, pos, qv, body, natom)
    eps4=4.0*eps_LJ; eps24=24.0*eps_LJ
    JACC.parallel_for(N, force_kernel!, Fv, Tv, Wbuf, Epbuf, pos, lab, nl_count, nl_list,
        h, hi, natom, MAX_NEIGH, rmcut2, sig2_LJ, eps4, eps24, VSHFT, RCUT2)
    Wbuf_h=Array(Wbuf); Epbuf_h=Array(Epbuf)
    Ep_ref[]=sum(Epbuf_h)
    for k in 1:9; s=0.0; for i in 1:N; s+=Wbuf_h[(i-1)*9+k]; end; Wm9[k]=s; end
    return nothing
end

function ke_trans(vel, N, Mmol)
    0.5*Mmol*JACC.parallel_reduce(N, ke_trans_kernel, vel)/CONV
end
function ke_rot(omg, N, I0)
    0.5*I0*JACC.parallel_reduce(N, ke_rot_kernel, omg)/CONV
end

function step_npt!(pos, vel, qv, omg, Fv, Tv, Wm9, Wbuf, Epbuf,
                   h, hi, body, I0, Mmol, N, natom, rmcut2, dt, npt, nl_count, nl_list, lab)
    hdt=0.5*dt; mat_inv9!(hi, h)
    V=abs(mat_det9(h)); kt=ke_trans(vel,N,Mmol); kr=ke_rot(omg,N,I0); KE=kt+kr
    npt.xi+=hdt*(2*KE-npt.Nf*kB*npt.Tt)/npt.Q; npt.xi=clamp(npt.xi,-0.1,0.1)
    dP=inst_P(Wm9,kt,V)-npt.Pe
    for a in 0:2; npt.Vg[a*4+1]+=hdt*V*dP/(npt.W*eV2GPa); npt.Vg[a*4+1]=clamp(npt.Vg[a*4+1],-0.01,0.01); end
    eps_tr=npt.Vg[1]*hi[1]+npt.Vg[5]*hi[5]+npt.Vg[9]*hi[9]
    sc_nh=exp(-hdt*npt.xi); sc_v=sc_nh*exp(-hdt*eps_tr/3.0)
    hdt_cF=hdt*CONV/Mmol; hdt_cT=hdt*CONV/I0
    JACC.parallel_for(N, vel_pre_kernel!, vel, omg, Fv, Tv, sc_v, sc_nh, hdt_cF, hdt_cT)
    JACC.parallel_for(N, pos_update_kernel!, pos, vel, hi, dt)
    for a in 0:2, b in 0:2; h[a*3+b+1]+=dt*npt.Vg[a*3+b+1]; end
    JACC.parallel_for(N, frac2cart_kernel!, pos, h)
    JACC.parallel_for(N, quat_update_kernel!, qv, omg, dt)
    mat_inv9!(hi, h)
    Ep_ref=Ref(0.0)
    compute_forces!(Fv,Tv,Wm9,Ep_ref,Wbuf,Epbuf,pos,qv,body,h,hi,nl_count,nl_list,lab,N,natom,rmcut2)
    eps_tr2=npt.Vg[1]*hi[1]+npt.Vg[5]*hi[5]+npt.Vg[9]*hi[9]
    sc_v2=sc_nh*exp(-hdt*eps_tr2/3.0)
    JACC.parallel_for(N, vel_post_kernel!, vel, omg, Fv, Tv, sc_v2, sc_nh, hdt_cF, hdt_cT)
    kt=ke_trans(vel,N,Mmol); kr=ke_rot(omg,N,I0); KE=kt+kr
    npt.xi+=hdt*(2*KE-npt.Nf*kB*npt.Tt)/npt.Q; npt.xi=clamp(npt.xi,-0.1,0.1)
    V2=abs(mat_det9(h)); dP=inst_P(Wm9,kt,V2)-npt.Pe
    for a in 0:2; npt.Vg[a*4+1]+=hdt*V2*dP/(npt.W*eV2GPa); npt.Vg[a*4+1]=clamp(npt.Vg[a*4+1],-0.01,0.01); end
    return (Ep_ref[], KE)
end

# ═══════════ Restart I/O ═══════════
function write_restart_lj(fname, istep, opts, st, nc, T, Pe, nsteps, dt, seed, fspec, init_scale,
                          h, npt, pos, qv, vel, omg, N, natom)
    open(fname, "w") do f
        print(f, "# RESTART fuller_LJ_npt_md_julia\n# OPTIONS:")
        for (k,v) in opts; print(f, " --$k=$v"); end
        @printf(f, "\nSTEP %d\nNSTEPS %d\nDT %.15e\nTEMP %.15e\nPRES %.15e\n", istep, nsteps, dt, T, Pe)
        @printf(f, "CRYSTAL %s\nNC %d\nFULLERENE %s\nINIT_SCALE %.15e\nSEED %d\n", st, nc, fspec, init_scale, seed)
        @printf(f, "NMOL %d\nNATOM_MOL %d\n", N, natom)
        print(f, "H"); for i in 1:9; @printf(f, " %.15e", h[i]); end; print(f, "\n")
        @printf(f, "NPT %.15e %.15e %.15e %.15e %.15e %d\n", npt.xi, npt.Q, npt.W, npt.Pe, npt.Tt, npt.Nf)
        print(f, "VG"); for i in 1:9; @printf(f, " %.15e", npt.Vg[i]); end; print(f, "\n")
        for i in 1:N
            @printf(f, "MOL %d %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
                i, pos[3i-2],pos[3i-1],pos[3i], qv[4i-3],qv[4i-2],qv[4i-1],qv[4i],
                vel[3i-2],vel[3i-1],vel[3i], omg[3i-2],omg[3i-1],omg[3i])
        end
        print(f, "END\n")
    end
end

function read_restart_lj(fname)
    h = zeros(9); Vg = zeros(9)
    xi=0.0; Q=0.0; W=0.0; Pe_r=0.0; Tt_r=0.0; Nf=0; istep=0; N_r=0
    pos_v=Float64[]; qv_v=Float64[]; vel_v=Float64[]; omg_v=Float64[]
    ok = false
    for line in readlines(fname)
        line = strip(line)
        if isempty(line) || line[1]=='#'; continue; end
        parts = split(line)
        tag = parts[1]
        if tag=="STEP"; istep=parse(Int,parts[2])
        elseif tag=="NMOL"; N_r=parse(Int,parts[2])
        elseif tag=="H"; for i in 1:9; h[i]=parse(Float64,parts[i+1]); end
        elseif tag=="NPT"
            xi=parse(Float64,parts[2]); Q=parse(Float64,parts[3]); W=parse(Float64,parts[4])
            Pe_r=parse(Float64,parts[5]); Tt_r=parse(Float64,parts[6]); Nf=parse(Int,parts[7])
        elseif tag=="VG"; for i in 1:9; Vg[i]=parse(Float64,parts[i+1]); end
        elseif tag=="MOL"
            vals = [parse(Float64, parts[i]) for i in 3:15]
            append!(pos_v, vals[1:3]); append!(qv_v, vals[4:7])
            append!(vel_v, vals[8:10]); append!(omg_v, vals[11:13])
        elseif tag=="END"; break; end
    end
    N_r = div(length(pos_v), 3); ok = N_r > 0
    npt = NPTState(xi, Q, Vg, W, Pe_r, Tt_r, Nf)
    if ok; @printf("  Restart loaded: %s (step %d, %d mols)\n", fname, istep, N_r); end
    return (istep=istep, N=N_r, h=h, npt=npt, pos=pos_v, qv=qv_v, vel=vel_v, omg=omg_v, ok=ok)
end

# ═══════════ MAIN ═══════════
function main()
    opts = parse_args(ARGS)
    if haskey(opts, "help")
        println("""fuller_LJ_npt_md.jl — LJ rigid-body fullerene NPT-MD (JACC.jl)

Options:
  --help                  Show this help
  --fullerene=<name>      Fullerene species (default: C60)
  --crystal=<fcc|hcp|bcc> Crystal structure (default: fcc)
  --cell=<nc>             Unit cell repeats (default: 3)
  --temp=<K>              Target temperature [K] (default: 298.0)
  --pres=<GPa>            Target pressure [GPa] (default: 0.0)
  --step=<N>              Production steps (default: 10000)
  --dt=<fs>               Time step [fs] (default: 1.0)
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
    dt = parse(Float64, get_opt(opts, "dt", "1.0"))
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
                 unique_file("ovito_traj_LJ_" * mode_tag * sfx, ".xyz")

    fpath, label = resolve_fullerene(fspec, libdir)
    mol = load_cc1(fpath)
    natom = mol.natom
    RMCUT = RCUT + 2*mol.Rmol + 1.0
    RMCUT2 = RMCUT*RMCUT
    a0 = default_a0(mol.Dmol > 0 ? mol.Dmol : 2*mol.Rmol, st, init_scale)

    Nmax = st=="FCC" ? 4*nc^3 : 2*nc^3

    # Allocate arrays
    pos=JACC.zeros(Float64,Nmax*3); vel=JACC.zeros(Float64,Nmax*3)
    omg=JACC.zeros(Float64,Nmax*3); qv=JACC.zeros(Float64,Nmax*4)
    Fv=JACC.zeros(Float64,Nmax*3); Tv=JACC.zeros(Float64,Nmax*3)
    lab=JACC.zeros(Float64,Nmax*natom*3)
    body=JACC.array(copy(mol.coords))
    h=zeros(Float64,9); hi=zeros(Float64,9); Wm9=zeros(Float64,9)
    Wbuf=JACC.zeros(Float64,Nmax*9); Epbuf=JACC.zeros(Float64,Nmax)
    nl_count=zeros(Int32,Nmax); nl_list=zeros(Int32,Nmax*MAX_NEIGH)

    # Build crystal
    pos_h=zeros(Nmax*3)
    if st=="FCC"; N=make_fcc!(pos_h,h,a0,nc)
    elseif st=="HCP"; N=make_hcp!(pos_h,h,a0,nc)
    else; N=make_bcc!(pos_h,h,a0,nc); end
    mat_inv9!(hi, h)
    copyto!(pos, JACC.array(pos_h))

    @printf("========================================================================\n")
    @printf("  Fullerene Crystal NPT-MD — LJ rigid-body (JACC.jl)\n")
    @printf("========================================================================\n")
    @printf("  Fullerene       : %s (%d atoms/mol)\n", label, natom)
    @printf("  Crystal         : %s %dx%dx%d  Nmol=%d\n", st, nc, nc, nc, N)
    @printf("  a0=%.3f A  T=%.1f K  P=%.4f GPa  dt=%.2f fs\n", a0, T, Pe, dt)
    if coldstart>0; @printf("  Coldstart       : %d steps at %.1f K\n", coldstart, T_cold); end
    if warmup>0; @printf("  Warmup          : %d steps (%.1fK->%.1fK)\n", warmup, T_cold, T); end
    @printf("  Production      : %d steps  avg=%d-%d\n", nsteps, avg_from, avg_to)
    @printf("  Total           : %d steps\n", total_steps)
    @printf("========================================================================\n\n")

    # Initial velocities
    T_init = (coldstart>0 || warmup>0) ? T_cold : T
    rng = MersenneTwister(seed)
    sv=sqrt(kB*T_init*CONV/mol.Mmol); sw=sqrt(kB*T_init*CONV/mol.I0)
    vel_h=zeros(N*3); omg_h=zeros(N*3); qv_h=zeros(N*4)
    for i in 1:N
        for a in 1:3; vel_h[3(i-1)+a]=sv*randn(rng); omg_h[3(i-1)+a]=sw*randn(rng); end
        for a in 1:4; qv_h[4(i-1)+a]=randn(rng); end
        n=sqrt(sum(qv_h[4(i-1)+a]^2 for a in 1:4)); for a in 1:4; qv_h[4(i-1)+a]/=n; end
    end
    vcm=zeros(3); for i in 1:N, a in 1:3; vcm[a]+=vel_h[3(i-1)+a]; end; vcm./=N
    for i in 1:N, a in 1:3; vel_h[3(i-1)+a]-=vcm[a]; end
    copyto!(vel, JACC.array(vel_h)); copyto!(omg, JACC.array(omg_h)); copyto!(qv, JACC.array(qv_h))

    npt = make_npt(T, Pe, N); npt.Tt = T_init
    start_step = 0

    # Load restart if specified
    if !isempty(resfile)
        rd = read_restart_lj(resfile)
        if rd.ok
            start_step = rd.istep; h .= rd.h; npt = rd.npt
            nn = min(N, rd.N)
            vel_h[1:nn*3] .= rd.vel[1:nn*3]; omg_h[1:nn*3] .= rd.omg[1:nn*3]
            pos_h[1:nn*3] .= rd.pos[1:nn*3]; qv_h[1:nn*4] .= rd.qv[1:nn*4]
            copyto!(pos, JACC.array(pos_h)); copyto!(vel, JACC.array(vel_h))
            copyto!(omg, JACC.array(omg_h)); copyto!(qv, JACC.array(qv_h))
            mat_inv9!(hi, h)
            @printf("  Restarting from global step %d\n", start_step)
        end
    end

    # Build initial neighbor list
    nlist_build_sym!(nl_count, nl_list, Array(pos), h, hi, N, RMCUT, MAX_NEIGH)
    nl_count_d = JACC.array(nl_count); nl_list_d = JACC.array(nl_list)

    # Initial PBC + forces
    h_d=JACC.array(h); hi_d=JACC.array(hi)
    JACC.parallel_for(N, pbc_kernel!, pos, h_d, hi_d)
    Ep_ref=Ref(0.0)
    compute_forces!(Fv,Tv,Wm9,Ep_ref,Wbuf,Epbuf,pos,qv,body,h_d,hi_d,nl_count_d,nl_list_d,lab,N,natom,RMCUT2)

    prn = mon_interval>0 ? mon_interval : max(1, div(total_steps, 50))
    prn_pre = prn
    if coldstart+warmup>0
        div_val = warmup_mon_mode=="freq" ? 10 : warmup_mon_mode=="some" ? 1000 : 100
        prn_pre = max(1, div(coldstart+warmup, div_val))
    end
    nlup=25; sT=0.0;sP=0.0;sa=0.0;sEp=0.0;nav=0
    t0=time()

    io_o = nrec_o>0 ? open(ovito_file, "w") : nothing

    @printf("  %8s %5s %7s %9s %8s %10s %13s %7s\n",
        "step","phase","T[K]","P[GPa]","a[A]","Ecoh[eV]","Ecoh[kcal/m]","t[s]")

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

        # Neighbor list rebuild
        if gstep%nlup==0
            pos_h_tmp=Array(pos); mat_inv9!(hi,h)
            nlist_build_sym!(nl_count,nl_list,pos_h_tmp,h,hi,N,RMCUT,MAX_NEIGH)
            nl_count_d=JACC.array(nl_count); nl_list_d=JACC.array(nl_list)
            h_d=JACC.array(h); hi_d=JACC.array(hi)
        end

        Ep,KE = step_npt!(pos,vel,qv,omg,Fv,Tv,Wm9,Wbuf,Epbuf,
            h_d,hi_d,body,mol.I0,mol.Mmol,N,natom,RMCUT2,dt,npt,nl_count_d,nl_list_d,lab)
        # Sync h back from device
        h .= Array(h_d); hi .= Array(hi_d)

        kt=ke_trans(vel,N,mol.Mmol); V=abs(mat_det9(h)); Tn=inst_T(KE,npt.Nf); Pn=inst_P(Wm9,kt,V)

        # Cold/warm velocity rescaling
        if (gstep<=coldstart || gstep<=coldstart+warmup) && Tn>0.1
            tgt = gstep<=coldstart ? T_cold : npt.Tt
            scale = sqrt(max(tgt,0.1)/Tn)
            JACC.parallel_for(N, vel_scale_kernel!, vel, omg, scale)
            kt=ke_trans(vel,N,mol.Mmol); KE=kt+ke_rot(omg,N,mol.I0); Tn=inst_T(KE,npt.Nf)
            npt.xi=0.0; if gstep<=coldstart; fill!(npt.Vg,0.0); end
        end

        Ec=Ep/N; an=h[1]/nc
        if gstep>=gavg_from && gstep<=gavg_to; sT+=Tn;sP+=Pn;sa+=an;sEp+=Ec;nav+=1; end

        # OVITO output
        if io_o!==nothing && gstep%nrec_o==0
            pos_h_o=Array(pos); vel_h_o=Array(vel); qv_h_o=Array(qv); body_h=Array(body)
            write_ovito_rigid(io_o, gstep, dt, pos_h_o, vel_h_o, qv_h_o, body_h, h, N, natom)
            flush(io_o)
        end

        # Restart save
        if nrec_rst>0 && (gstep%nrec_rst==0 || gstep==total_steps)
            rfn = restart_filename(ovito_file, gstep, total_steps)
            pos_h_r=Array(pos);vel_h_r=Array(vel);omg_h_r=Array(omg);qv_h_r=Array(qv)
            write_restart_lj(rfn,gstep,opts,st,nc,T,Pe,nsteps,dt,seed,fspec,init_scale,
                h,npt,pos_h_r,qv_h_r,vel_h_r,omg_h_r,N,natom)
            if stop_requested
                @printf("\n  *** Stopped at restart checkpoint (step %d) ***\n", gstep); break
            end
        end

        # Monitor + stop control
        if gstep%cur_prn==0 || gstep==total_steps
            if dir_exists("abort.md")
                @printf("\n  *** abort.md detected at step %d ***\n", gstep)
                if nrec_rst>0
                    rfn=restart_filename(ovito_file,gstep,total_steps)
                    pos_h_r=Array(pos);vel_h_r=Array(vel);omg_h_r=Array(omg);qv_h_r=Array(qv)
                    write_restart_lj(rfn,gstep,opts,st,nc,T,Pe,nsteps,dt,seed,fspec,init_scale,
                        h,npt,pos_h_r,qv_h_r,vel_h_r,omg_h_r,N,natom)
                end
                break
            end
            if !stop_requested && dir_exists("stop.md")
                stop_requested=true
                @printf("\n  *** stop.md detected at step %d — will stop at next checkpoint ***\n", gstep)
            end
            el=time()-t0
            @printf("  %8d %5s %7.1f %9.3f %8.3f %10.5f %13.4f %7.0f\n",
                gstep,phase,Tn,Pn,an,Ec,Ec*eV2kcalmol,el)
        end
    end

    if io_o!==nothing; close(io_o); end
    if nav>0
        @printf("\n========================================================================\n")
        @printf("  Averages (%d samples): T=%.2f K  P=%.4f GPa  a=%.4f A  Ecoh=%.5f eV\n",
            nav, sT/nav, sP/nav, sa/nav, sEp/nav)
        @printf("========================================================================\n")
    end
    @printf("  Done (%.1f sec)\n", time()-t0)
    return 0
end

main()
