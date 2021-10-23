# This file defines the various types of scores we need for
# lensing reconstructions
import jax
import jax.numpy as jnp
from jax_lensing.inversion import ks93inv

def log_gaussian_prior(x, sigma, ps_map):
  """ Gaussian prior on the power spectrum of the map.

  Args:
    x: tensor of shape (nx, nx, 2)
    sigma: scalar temperature
    ps_map: map of the power spectrum, of shape (nx, nx)

  Returns:
    log prior
  """
  data_ft = jnp.fft.fft2(x, axes=[0,1]) / x.shape[0]
  prior_E = -0.5*jnp.sum(jnp.real(data_ft[...,0]*jnp.conj(data_ft[...,0])) / (ps_map+sigma**2))
  prior_B = -0.5*jnp.sum(jnp.real(data_ft[...,1]*jnp.conj(data_ft[...,1])) / (sigma**2))
  return prior_E + prior_B

gaussian_prior_score = jax.vmap(jax.grad(log_gaussian_prior), in_axes=[0,0, None])

def log_likelihood(x, sigma, meas_shear, sigma_gamma):
  """ Likelihood function at the level of the measured shear
  Args:
    x: tensor of shape (nx, nx, 2) of inputt convergence
    sigma: scalar temperature
    meas_shear: tensor of shape (nx, nx, 2) of (g1,g2)
    sigma_gamma: variance map of input shear

  Returns:
    log likelihood
  """
  model_shear = jnp.stack(ks93inv(x[...,0], x[...,1]), axis=-1)  
  return - 0.5*jnp.sum((model_shear - meas_shear)**2/(sigma_gamma**2 + sigma**2))

likelihood_score = jax.vmap(jax.grad(log_likelihood), in_axes=[0,0, None, None])



