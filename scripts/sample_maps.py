# Script for sampling constrained realisations
import os
# This line is for running on Jean Zay
#os.environ['XLA_FLAGS']='--xla_gpu_cuda_data_dir=/gpfslocalsys/cuda/10.0'
os.environ['XLA_FLAGS']='--xla_gpu_cuda_data_dir=/gpfslocalsys/cuda/10.1.2'

from absl import app
from absl import flags
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as onp
import pickle
from functools import partial
from astropy.io import fits

# Import tensorflow for dataset creation and manipulation
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

from jax_lensing.models.convdae2 import SmallUResNet, MediumUResNet
from jax_lensing.models.normalization import SNParamsTree as CustomSNParamsTree
from jax_lensing.spectral import make_power_map
from jax_lensing.inversion import ks93inv, ks93

import tensorflow_probability as tfp; tfp = tfp.experimental.substrates.jax
from jax_lensing.samplers.score_samplers import ScoreHamiltonianMonteCarlo
from jax_lensing.samplers.tempered_sampling import TemperedMC

flags.DEFINE_string("output_file", "samples.fits", "Filename of output samples.")
flags.DEFINE_string("shear", "gamma.fits", "Path to input shear maps.")
flags.DEFINE_string("mask", "mask.fits", "Path to input mask.")
flags.DEFINE_float("sigma_gamma", 0.15, "Standard deviation of shear.")
flags.DEFINE_string("model_weights", "model-final.pckl", "Path to trained model weights.")
flags.DEFINE_string("variant", "EiffL", "Variant of model.")
flags.DEFINE_string("model", "SmallUResNet", "Name of model.")
flags.DEFINE_integer("batch_size", 32, "Size of batch to sample in parallel.")
flags.DEFINE_float("initial_temperature", 0.15, "Initial temperature at which to start sampling.")
flags.DEFINE_float("initial_step_size", 0.01, "Initial step size at which to perform sampling.")
flags.DEFINE_integer("min_steps_per_temp", 10, "Minimum number of steps for each temperature.")
flags.DEFINE_integer("num_steps", 5000, "Total number of steps in the chains.")
flags.DEFINE_integer("output_steps", 1, "How many steps to output.")
flags.DEFINE_string("gaussian_path", "data/massivenu/mnu0.0_Maps10_PS_theory.npy", "Path to Massive Nu power spectrum.")
flags.DEFINE_boolean("gaussian_only", False, "Only use Gaussian score if yes.")
flags.DEFINE_boolean("reduced_shear", False, "Apply reduced shear correction if yes.")
flags.DEFINE_float("resolution", 0.29, "Map resoultion arcmin/pixel")

FLAGS = flags.FLAGS

def forward_fn(x, s, is_training=False):
  if FLAGS.model == 'SmallUResNet':
    denoiser = SmallUResNet(n_output_channels=1, variant=FLAGS.variant)
  elif FLAGS.model == 'MediumUResNet':
    denoiser = MediumUResNet()
  else:
    raise NotImplementedError
  return denoiser(x, s, is_training=is_training)

def log_gaussian_prior(map_data, sigma, ps_map):
  """ Gaussian prior on the power spectrum of the map
  """
  data_ft = jnp.fft.fft2(map_data) / map_data.shape[0]
  return -0.5*jnp.sum(jnp.real(data_ft*jnp.conj(data_ft)) / (ps_map+sigma**2))
gaussian_prior_score = jax.vmap(jax.grad(log_gaussian_prior), in_axes=[0,0, None])

def log_likelihood(x, sigma, meas_shear, mask):
  """ Likelihood function at the level of the measured shear
  """
  ke = x.reshape((mask.shape[0], mask.shape[0]))
  kb = jnp.zeros(ke.shape)
  model_shear = jnp.stack(ks93inv(ke, kb), axis=-1)
  if FLAGS.reduced_shear:
    model_shear = model_shear /( 1. - jnp.clip(jnp.expand_dims(ke,axis=-1), -1., 0.9))
  return - jnp.sum(mask*(model_shear - meas_shear)**2/((FLAGS.sigma_gamma)**2 + sigma**2) )/2.
likelihood_score = jax.vmap(jax.grad(log_likelihood), in_axes=[0,0, None, None])

def main(_):

  map_size = fits.getdata(FLAGS.mask).astype('float32').shape[0] 

  # Make the network
  model = hk.transform_with_state(forward_fn)
  rng_seq = hk.PRNGSequence(42)
  params, state = model.init(next(rng_seq),
                             jnp.zeros((1, map_size, map_size, 2)),
                             jnp.zeros((1, 1, 1, 1)), is_training=True)

  # Load the weights of the neural network
  if not FLAGS.gaussian_only:
    with open(FLAGS.model_weights, 'rb') as file:
      params, state, sn_state = pickle.load(file)
    residual_prior_score = partial(model.apply, params, state, next(rng_seq), is_training=True)

  pixel_size = jnp.pi * FLAGS.resolution / 180. / 60. #rad/pixel
  # Load prior power spectrum
  ps_data = onp.load(FLAGS.gaussian_path).astype('float32')
  ell = jnp.array(ps_data[0,:])
  # 4th channel for massivenu
  ps_halofit = jnp.array(ps_data[1,:] / pixel_size**2) # normalisation by pixel size
  # convert to pixel units of our simple power spectrum calculator
  #kell = ell / (360/3.5/0.5) / float(map_size)
  kell = ell /2/jnp.pi * 360 * pixel_size / map_size
  # Interpolate the Power Spectrum in Fourier Space
  power_map = jnp.array(make_power_map(ps_halofit, map_size, kps=kell))

  # Load the shear maps and corresponding mask
  gamma = fits.getdata(FLAGS.shear).astype('float32') # Shear is expected in the format [map_size,map_size,2]
  mask = jnp.expand_dims(fits.getdata(FLAGS.mask).astype('float32'), -1) # has shape [map_size,map_size,1]

  @jax.jit
  def total_score_fn(x, sigma):
    """ Compute the total score, combining the following components:
        gaussian prior, ml prior, data likelihood
    """
    x = x.reshape([FLAGS.batch_size, map_size, map_size])
    # Retrieve Gaussian score
    gaussian_score = gaussian_prior_score(x, sigma, power_map)
    if FLAGS.gaussian_only:
      ml_score = 0
    else:
      # Use Neural network to compute residual prior score
      net_input = jnp.stack([x, sigma.reshape([-1,1,1])**2 * gaussian_score], axis=-1)
      ml_score = residual_prior_score(net_input, sigma.reshape([-1,1,1,1]))[0][...,0]
    # Compute likelihood score
    data_score = likelihood_score(x, sigma, gamma, mask)
    return (data_score + gaussian_score + ml_score).reshape((-1,map_size*map_size))

  # Prepare the first guess convergence by adding noise in the holes and performing
  # a KS inversion
  gamma_init = (jnp.repeat(jnp.expand_dims(mask*gamma,0), FLAGS.batch_size, axis=0) +
                jnp.expand_dims((1. - mask)*FLAGS.sigma_gamma,0)*onp.random.randn(FLAGS.batch_size, map_size, map_size, 2))
  kappa_init, _ = jax.vmap(ks93)(gamma_init[...,0], gamma_init[...,1])
  # we only care about kappa_e

  # Adds further noise in the image if the initial temp is above sigma_gamma
  if FLAGS.initial_temperature > FLAGS.sigma_gamma:
    delta_tmp = onp.sqrt(FLAGS.initial_temperature**2 - FLAGS.sigma_gamma**2)
    kappa_init = kappa_init + delta_tmp*onp.random.randn(FLAGS.batch_size, map_size,map_size)
  print(kappa_init.shape)
  # And now we can sample
  def make_kernel_fn(target_log_prob_fn, target_score_fn, sigma):
    return ScoreHamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        target_score_fn=target_score_fn,
        step_size=FLAGS.initial_step_size*(jnp.max(sigma)/FLAGS.initial_temperature)**0.5,
        num_leapfrog_steps=3,
        num_delta_logp_steps=4)

  tmc = TemperedMC(
        target_score_fn=total_score_fn,
        inverse_temperatures=FLAGS.initial_temperature*jnp.ones([FLAGS.batch_size]),
        make_kernel_fn=make_kernel_fn,
        gamma=0.98,
        min_steps_per_temp=FLAGS.min_steps_per_temp,
        num_delta_logp_steps=4)

  samples, trace = tfp.mcmc.sample_chain(
          num_results=FLAGS.output_steps,
          current_state=kappa_init.reshape([FLAGS.batch_size,-1]),
          kernel=tmc,
          num_burnin_steps=0,
          num_steps_between_results=FLAGS.num_steps//FLAGS.output_steps,
          trace_fn=lambda _, pkr: (pkr.pre_tempering_results.is_accepted,
                                   pkr.post_tempering_inverse_temperatures,
                                   pkr.tempering_log_accept_ratio),
          seed=jax.random.PRNGKey(onp.random.randint(0,10000)))

  print('average acceptance rate', onp.mean(trace[0]))
  print('final max temperature', onp.max(trace[1][:,-1]))
  # TODO: apply final projection
  # Save the chain
  fits.writeto(FLAGS.output_file, onp.array(samples),overwrite=True)

if __name__ == "__main__":
  app.run(main)
