# jax-lensing

<<<<<<< master
[![arXiv:2011.08271](https://img.shields.io/badge/astro--ph.CO-arXiv%3A2011.08271-B31B1B.svg)](https://arxiv.org/abs/2011.08271) [![License](https://img.shields.io/pypi/l/jax-cosmo)](https://github.com/CosmoStat/jax-lensing/tree/master/LICENSE)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-4-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
A JAX package for gravitational lensing

This repository contains scripts and notebook to reproduce the results from the following paper:

[Probabilistic Mass-Mapping with Neural Score Estimation](https://arxiv.org/abs/2011.08271), B. Remy, F. Lanusse, N. Jeffrey, J. Liu, J.-L. Starck, K. Osato, T. Schrabback, submitted to Astronomy and Astrophisics, 2021.

You will find all the scripts and instructions to reproduce traine models, sample maps and reproduce the paper figures [here](https://github.com/CosmoStat/jax-lensing/tree/master/papers/Remy2021).

This repository also contains implementation of the following methods:
- [`DeepLensingPosterior`](https://arxiv.org/abs/2011.08271)
- [`DeepMass`](https://arxiv.org/abs/1908.00543v2)
- [`Wiener Filter`](https://www.aanda.org/articles/aa/abs/2013/01/aa20586-12/aa20586-12.html)
- [`Kaiser-Squires`](https://ui.adsabs.harvard.edu/abs/1993ApJ...404..441K/abstract)

## Convergence posterior sampling
This package enables to sample **high resolution convergence map** from the posterior distribution in 10 GPU-minutes (on an Nvidia Tesla V100 GPU) in average. Have a look at the *annealed Hamiltonian Monte Carlo* sampling scheme bellow:

<img src="assets/video-annealing.gif">

## Mass-mapping
Comparison between
`DLPosterior`, `DeepMass`, `Wiener Filter` and `KS93` methods.

| Ground truth convergence                      | `DLPosterior` samples                     |
|-----------------------------------------------|-------------------------------------------|
| <img height=300 src="assets/convergence.png"> | <img height=300 src="assets/cropped.gif"> |

| `DeepMass`                                    | `DLPosterior` mean                        |
|-----------------------------------------------|-------------------------------------------|
| <img height=300 src="assets/deepmass.png">    | <img height=300 src="assets/dlp.png">     |

| `Wiener Filter`                               | `Kaiser-Squires`                          |
|-----------------------------------------------|-------------------------------------------|
| <img height=300 src="assets/wiener.png">      | <img height=300 src="assets/ks.png">      |


## Install

`jax-lensing` is pure python and can be easily installed with `pip`:

```
$ cd jax-lensing
$ pip install .
```

## Requirements
- [Jax](https://github.com/google/jax) [==0.2.18]
- [Haiku](https://github.com/deepmind/dm-haiku) [==0.0.4]
- [Astropy](https://github.com/astropy/astropy) [==4.2]
- [TensorFlow Probability](https://github.com/tensorflow/probability) [==0.13.0]
- [TensorFlow Datasets](https://github.com/tensorflow/datasets) [==4.3.0]
- [TensorFlow](https://www.tensorflow.org/) [==2.5.0]

## Citation

If you use `jax-lensing` in a scientific publication, we would appreciate citations to the following paper:

[Probabilistic Mass-Mapping with Neural Score Estimation](https://arxiv.org/abs/2011.08271), B. Remy, F. Lanusse, N. Jeffrey, J. Liu, J.-L. Starck, K. Osato, T. Schrabback, submitted to Astronomy and Astrophisics, 2021.


The BibTeX citation is the following:
```
@Upcomming
```

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://www.cosmostat.org/people/benjamin-remy"><img src="https://avatars.githubusercontent.com/u/30293694?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Benjamin Remy</b></sub></a><br /><a href="#infra-b-remy" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/CosmoStat/jax-lensing/commits?author=b-remy" title="Tests">⚠️</a> <a href="https://github.com/CosmoStat/jax-lensing/commits?author=b-remy" title="Code">💻</a></td>
    <td align="center"><a href="http://flanusse.net"><img src="https://avatars.githubusercontent.com/u/861591?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Francois Lanusse</b></sub></a><br /><a href="#infra-EiffL" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/CosmoStat/jax-lensing/commits?author=EiffL" title="Tests">⚠️</a> <a href="https://github.com/CosmoStat/jax-lensing/commits?author=EiffL" title="Code">💻</a></td>
    <td align="center"><a href="https://nialljeffrey.github.io/"><img src="https://avatars.githubusercontent.com/u/15345794?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Niall Jeffrey</b></sub></a><br /><a href="https://github.com/CosmoStat/jax-lensing/commits?author=NiallJeffrey" title="Tests">⚠️</a> <a href="https://github.com/CosmoStat/jax-lensing/commits?author=NiallJeffrey" title="Code">💻</a></td>
    <td align="center"><a href="https://liuxx479.github.io/"><img src="https://avatars.githubusercontent.com/u/2658222?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jia Liu</b></sub></a><br /><a href="https://github.com/CosmoStat/jax-lensing/commits?author=liuxx479" title="Tests">⚠️</a> <a href="https://github.com/CosmoStat/jax-lensing/commits?author=liuxx479" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
=======
>>>>>>> docs