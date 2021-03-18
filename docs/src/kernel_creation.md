# GLOM Kernel Creation

`GLOM` requires latent GP kernels that are at least twice mean-square differentiable in order to ensure that second order derivatives exist for non-trivial model construction. All of the derived versions of the kernels that are necessary can be created using the following function.

```@autodocs
Modules = [GPLinearODEMaker]
Pages   = ["src/kernel_creation_functions.jl"]
```

All of the premade kernels that are included with `GLOM` (in `src/kernels`) were created with this [example script](https://github.com/christiangil/GPLinearODEMaker.jl/blob/master/examples/creating_kernels.jl)
