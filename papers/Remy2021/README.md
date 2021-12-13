# Notebooks and support scripts for Remy et al. 2021

This folder contains all of the data download instructions, support scripts,
and analysis notebooks needed to reproduce the results of Remy et al. 2021.

## Data Download and Preparation


## Training a Score Network
```
$ sbatch score_trainin_jz.job
```

## Posterior sampling

Sampling posterior with COSMOS input shear
```
$ sbatch sample_COSMOS_jz.job
```

Sampling posterior with simulated input shear
```
$ sbatch sample_simulations_jz.job
```

Sampling posterior with simulated input shear, with added cluster
```
$ sbatch sample_with_nfw_jz.job
```

## Training DeepMass 
```
$ sbatch deepmass_training_jz.job
```

## Analysis notebooks

- [COSMOS_shape_catalog.ipynb](https://github.com/CosmoStat/jax-lensing/blob/master/papers/Remy2021/COSMOS_shape_catalog.ipynb): Data processing.

- [MassMappingResults.ipynb](https://github.com/CosmoStat/jax-lensing/blob/master/papers/Remy2021/MassMappingResults.ipynb): Visualize `DLPosterior` mean and compare to other methods.

- [WienerGaussianPrior.ipynb](https://github.com/CosmoStat/jax-lensing/blob/master/papers/Remy2021/WienerGaussianPrior.ipynb): Comparing method to compute the Wiener Filter

- [HMCSampling.ipynb](https://github.com/CosmoStat/jax-lensing/blob/master/papers/Remy2021/HMCSampling.ipynb): Detailed *annealed Hamiltonian Motecarlo* procedure.

- [HybridDenoiser.ipynb](https://github.com/CosmoStat/jax-lensing/blob/master/papers/Remy2021/HybridDenoiser.ipynb): Turning the score function into a denoiser and demonstrating the hybrid Gaussian-Residual denoiser method.

- [SpectralNormRegularization.ipynb](https://github.com/CosmoStat/jax-lensing/blob/master/papers/Remy2021/SpectralNormRegularization.ipynb): Demonstration of the regularization with Spectral Normalization when learning the score function.

- [Detection.ipynb](https://github.com/CosmoStat/jax-lensing/blob/master/papers/Remy2021/Detection.ipynb): Detection experiment from posterior samples.

- [DeepMass.ipynb](https://github.com/CosmoStat/jax-lensing/blob/master/papers/Remy2021/DeepMass.ipynb): Inference with `DeepMass`.