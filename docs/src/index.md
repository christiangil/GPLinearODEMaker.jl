```@meta
CurrentModule = GPLinearODEMaker
```

# GPLinearODEMaker.jl Documentation

GPLinearODEMaker (GLOM) is a package for finding the likelihood (and derivatives thereof) of multivariate Gaussian processes (GP) that are composed of a linear combination of a univariate GP and its derivatives.

![q_0(t) = m_0(t) + a_{00}X(t) + a_{01}\dot{X}(t) + a_{02}\ddot{X}(t)](https://render.githubusercontent.com/render/math?math=q_0(t)%20%3D%20m_0(t)%20%2B%20a_%7B00%7DX(t)%20%2B%20a_%7B01%7D%5Cdot%7BX%7D(t)%20%2B%20a_%7B02%7D%5Cddot%7BX%7D(t))

![q_1(t) = m_1(t) + a_{10}X(t) + a_{11}\dot{X}(t) + a_{12}\ddot{X}(t)](https://render.githubusercontent.com/render/math?math=q_1(t)%20%3D%20m_1(t)%20%2B%20a_%7B10%7DX(t)%20%2B%20a_%7B11%7D%5Cdot%7BX%7D(t)%20%2B%20a_%7B12%7D%5Cddot%7BX%7D(t))

![\vdots](https://render.githubusercontent.com/render/math?math=%5Cvdots)

![q_l(t) = m_l(t) + a_{l0}X(t) + a_{l1}\dot{X}(t) + a_{l2}\ddot{X}(t)](https://render.githubusercontent.com/render/math?math=q_l(t)%20%3D%20m_l(t)%20%2B%20a_%7Bl0%7DX(t)%20%2B%20a_%7Bl1%7D%5Cdot%7BX%7D(t)%20%2B%20a_%7Bl2%7D%5Cddot%7BX%7D(t))

where each X(t) is the latent GP and the qs are the time series of the outputs.

## Where to begin?

If you haven't used GLOM before, a good place to start is the "Getting Started" section. We list how to install the package as well as a simple example

```@contents
Pages = ["gettingstarted.md"]
Depth = 2
```

## User's Guide

Using `GLOM` generally starts with choosing a kernel function and creating a [`GLO`](@ref) object. Several kernel functions have been created already and are stored in `src/kernels`

```@contents
Pages = ["kernel.md"]
Depth = 2
```

```@contents
Pages = ["glo.md"]
Depth = 2
```

```@contents
Pages = ["kernel_creation.md"]
Depth = 2
```



```@contents
Pages = ["nlogl.md"]
Depth = 2
```

```@contents
Pages = ["priors.md"]
Depth = 2
```

## Citing GLOM

If you use `GPLinearODEMaker.jl` in your work, please cite the following BibTeX entry

```
@ARTICLE{2020ApJ...905..155G,
       author = {{Gilbertson}, Christian and {Ford}, Eric B. and {Jones}, David E. and {Stenning}, David C.},
        title = "{Toward Extremely Precise Radial Velocities. II. A Tool for Using Multivariate Gaussian Processes to Model Stellar Activity}",
      journal = {\apj},
     keywords = {Exoplanet detection methods, Astronomy software, Stellar activity, Gaussian Processes regression, Time series analysis, 489, 1855, 1580, 1930, 1916, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Solar and Stellar Astrophysics},
         year = 2020,
        month = dec,
       volume = {905},
       number = {2},
          eid = {155},
        pages = {155},
          doi = {10.3847/1538-4357/abc627},
archivePrefix = {arXiv},
       eprint = {2009.01085},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020ApJ...905..155G},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
## Indices

All of the package functions and types can be found here

```@contents
Pages = ["indices.md"]
```

## Documentation Acknowledgments
Thanks to [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) for making Julia documentation easier and [Augmentor.jl](https://github.com/Evizero/Augmentor.jl) for documentation inspiration.
