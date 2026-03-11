#!/usr/bin/env julia
# One-time setup script for fuller_md_Julia
# Usage: julia setup.jl [backend]
#   backend: threads (default), cuda, amdgpu, oneapi

import Pkg
Pkg.activate(@__DIR__)
Pkg.add("JACC")

backend = length(ARGS) >= 1 ? ARGS[1] : "threads"
println("Setting JACC backend to: $backend")

import JACC
JACC.set_backend(backend)

if backend == "cuda"
    Pkg.add("CUDA")
elseif backend == "amdgpu"
    Pkg.add("AMDGPU")
elseif backend == "oneapi"
    Pkg.add("oneAPI")
end

Pkg.instantiate()
println("Setup complete. Backend: $backend")
