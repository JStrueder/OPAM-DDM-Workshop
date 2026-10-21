# Drift Diffusion Models: a practical introduction

Materials for the OPAM 2026 workshop by Jeremy Strueder (IESE Business School).

The R scripts here run the worked example from the slides: speed vs. accuracy instructions in lexical decision (Wagenmakers, Ratcliff, Gomez, & McKoon, 2008, Experiment 1).

## Files

| File | What it does | Needs |
|---|---|---|
| `driftd_workshop.R` | **Start here.** The full analysis from the slides: data, model specification, fitting, convergence, estimates, effects, and model checks. Fits each participant separately by maximum likelihood with [driftd](https://github.com/JStrueder/driftd). | R 3.6 or later. About 10–15 minutes on a laptop. |
| `driftd_workshop_brms.R` | Optional. The same analysis as a hierarchical Bayesian model with [brms](https://paulbuerkner.com/brms/). | R 4.0 or later and a working Stan installation (rstan or cmdstanr). The model fit takes about 1.5 hours with 4 chains on 4 cores. |

## How to run

1. Download `driftd_workshop.R` (or clone this repository) and open it in R or RStudio.
2. Run it from top to bottom, or section by section along with the slides.

The first section installs driftd from GitHub if you don't have it yet. You can also install it yourself beforehand:

```r
install.packages("remotes")
remotes::install_github("JStrueder/driftd@v0.1.0", build = FALSE)
```

This needs an internet connection the first time. No Rtools or compiler is needed.

## Data

The data come with driftd (`data(speed_acc)`), so there is nothing else to download. They are Experiment 1 of Wagenmakers et al. (2008), as distributed with the rtdists package.

## References

Wagenmakers, E.-J., Ratcliff, R., Gomez, P., & McKoon, G. (2008). A diffusion model account of criterion shifts in the lexical decision task. *Journal of Memory and Language, 58*, 140–159.
