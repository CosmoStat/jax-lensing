# Script for sampling constrained realisations
import os
# This line is for running on Jean Zay
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

from jax_lensing.models.convdae import SmallUResNet
from jax_lensing.models.convdae2 import MediumUResNet
from jax_lensing.models.normalization import SNParamsTree as CustomSNParamsTree
from jax_lensing.spectral import make_power_map
from jax_lensing.inversion import ks93inv, ks93
from jax_lensing.cluster import gen_nfw_shear
from jax_lensing.samplers.procedures import tempered_HMC

import tensorflow_probability as tfp; tfp = tfp.experimental.substrates.jax

flags.DEFINE_string("output_folder", "100_100_.5_3e14_1", "Name of the output folder.")
flags.DEFINE_string("output_file", "samples.fits", "Filename of output samples.")
#flags.DEFINE_string("shear", "gamma.fits", "Path to input shear maps.")
flags.DEFINE_string("convergence", "convergence.npy", "Path to input noiseless convergence maps.")
flags.DEFINE_string("mask", "mask.fits", "Path to input mask.")
flags.DEFINE_float("sigma_gamma", 0.15, "Standard deviation of shear.")
flags.DEFINE_string("model_weights", "model-final.pckl", "Path to trained model weights.")
flags.DEFINE_string("variant", "EiffL", "Variant of model.")
flags.DEFINE_string("model", "SmallUResNet", "Name of model.")
flags.DEFINE_integer("batch_size", 5, "Size of batch to sample in parallel.")
flags.DEFINE_float("initial_temperature", 0.15, "Initial temperature at which to start sampling.")
flags.DEFINE_float("initial_step_size", 0.01, "Initial step size at which to perform sampling.")
flags.DEFINE_integer("min_steps_per_temp", 10, "Minimum number of steps for each temperature.")
flags.DEFINE_integer("num_steps", 5000, "Total number of steps in the chains.")
flags.DEFINE_integer("output_steps", 3, "How many steps to output.")
flags.DEFINE_string("gaussian_path", "data/massivenu/mnu0.0_Maps10_PS_theory.npy", "Path to Massive Nu power spectrum.")
flags.DEFINE_boolean("gaussian_only", False, "Only use Gaussian score if yes.")
flags.DEFINE_boolean("reduced_shear", False, "Apply reduced shear correction if yes.")
flags.DEFINE_integer("map_size", 360, "Map size")
flags.DEFINE_float("resolution", 0.29, "Map resoultion arcmin/pixel")
flags.DEFINE_boolean("no_cluster", False, "No added cluster if True")
flags.DEFINE_integer("x_cluster", 100, "x-coordinate of the cluster")
flags.DEFINE_integer("y_cluster", 100, "y-coordinate of the cluster")
flags.DEFINE_float("z_halo", .5, "redshift of the cluster")
flags.DEFINE_float("mass_halo", 2e15, "mass of the cluster (in solar mass)")
flags.DEFINE_float("zs", 1, "redshif of the source")
flags.DEFINE_boolean("COSMOS", False, "COSMOS catalog")

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
  data_ft = jnp.fft.fft2(map_data) / FLAGS.map_size
  return -0.5*jnp.sum(jnp.real(data_ft*jnp.conj(data_ft)) / (ps_map+sigma**2))
gaussian_prior_score = jax.vmap(jax.grad(log_gaussian_prior), in_axes=[0,0, None])

def log_likelihood(x, sigma, meas_shear, mask):
  """ Likelihood function at the level of the measured shear
  """
  ke = x.reshape((FLAGS.map_size, FLAGS.map_size))
  kb = jnp.zeros(ke.shape)
  model_shear = jnp.stack(ks93inv(ke, kb), axis=-1)
  if FLAGS.reduced_shear:
    model_shear = model_shear /( 1. - jnp.clip(jnp.expand_dims(ke,axis=-1), -1., 0.9))
  return - jnp.sum(mask*(model_shear - meas_shear)**2/((FLAGS.sigma_gamma)**2 + sigma**2) )/2.
likelihood_score = jax.vmap(jax.grad(log_likelihood), in_axes=[0,0, None, None])

def main(_):
  # Make the network
  model = hk.transform_with_state(forward_fn)
  rng_seq = hk.PRNGSequence(42)
  params, state = model.init(next(rng_seq),
                             jnp.zeros((1, FLAGS.map_size, FLAGS.map_size, 2)),
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
  #kell = ell / (360/3.5/0.5) / float(FLAGS.map_size)
  kell = ell /2/jnp.pi * 360 * pixel_size / FLAGS.map_size
  # Interpolate the Power Spectrum in Fourier Space
  power_map = jnp.array(make_power_map(ps_halofit, FLAGS.map_size, kps=kell))


  # Load the noiseless convergence map
  if not FLAGS.COSMOS:
    print('i am here')
    convergence= fits.getdata(FLAGS.convergence).astype('float32')
    #convergence = onp.load(FLAGS.convergence).astype('float32') #Convergence is expected in the format [FLAGS.map_size,FLAGS.map_size,1]
  
    # Get the correspinding shear
    gamma1, gamma2 = ks93inv(convergence, onp.zeros_like(convergence)) 
  
    if not FLAGS.no_cluster: 
      print('adding a cluster')
      # Compute NFW profile shear map
      g1_NFW, g2_NFW = gen_nfw_shear(x_cen=FLAGS.x_cluster, y_cen=FLAGS.y_cluster, 
                                   resolution=FLAGS.resolution,
                                   nx=FLAGS.map_size, ny=FLAGS.map_size, z=FLAGS.z_halo,
                                   m=FLAGS.mass_halo, zs=FLAGS.zs)
      # Shear with added NFW cluster
      gamma1 += g1_NFW
      gamma2 += g2_NFW
  

      # Target convergence map with the added cluster
      #ke_cluster, kb_cluster = ks93(g1_cluster, g2_cluster)

    # Add noise the shear map
    gamma1 += FLAGS.sigma_gamma * onp.random.randn(FLAGS.map_size,FLAGS.map_size)
    gamma2 += FLAGS.sigma_gamma * onp.random.randn(FLAGS.map_size,FLAGS.map_size)


    # Load the shear maps and corresponding mask
    gamma = onp.stack([gamma1, gamma2], -1) # Shear is expected in the format [FLAGS.map_size,FLAGS.map_size,2]
    #mask = jnp.expand_dims(onp.ones_like(gamma1), -1) # has shape [FLAGS.map_size,FLAGS.map_size,1]

    #gamma = fits.getdata(FLAGS.shear).astype('float32') # Shear is expected in the format [FLAGS.map_size,FLAGS.map_size,2]
    #mask = jnp.expand_dims(fits.getdata(FLAGS.mask).astype('float32'), -1) # has shape [FLAGS.map_size,FLAGS.map_size,1]

  else:

    # Load the shear maps and corresponding mask
    g1 = fits.getdata('../data/COSMOS/cosmos_full_e1_0.29arcmin360.fits').astype('float32').reshape([FLAGS.map_size, FLAGS.map_size, 1])
    g2 = fits.getdata('../data/COSMOS/cosmos_full_e2_0.29arcmin360.fits').astype('float32').reshape([FLAGS.map_size, FLAGS.map_size, 1])
    gamma = onp.concatenate([g1, g2], axis=-1)

  mask = jnp.expand_dims(fits.getdata(FLAGS.mask).astype('float32'), -1) # has shape [FLAGS.map_size,FLAGS.map_size,1]

  @jax.jit
  def total_score_fn(x, sigma):
    """ Compute the total score, combining the following components:
        gaussian prior, ml prior, data likelihood
    """
    x = x.reshape([FLAGS.batch_size, FLAGS.map_size, FLAGS.map_size])
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
    return (data_score + gaussian_score + ml_score).reshape((-1,FLAGS.map_size*FLAGS.map_size))


  # Prepare the first guess convergence by adding noise in the holes and performing
  # a KS inversion
  print(gamma.shape)
  print(jnp.repeat(jnp.expand_dims(mask*gamma,0), FLAGS.batch_size, axis=0).shape)
  print((jnp.expand_dims((1. - mask)*FLAGS.sigma_gamma,0)*onp.random.randn(FLAGS.batch_size, FLAGS.map_size, FLAGS.map_size, 2)).shape)
  gamma_init = (jnp.repeat(jnp.expand_dims(mask*gamma,0), FLAGS.batch_size, axis=0) +
                
                jnp.expand_dims((1. - mask)*FLAGS.sigma_gamma,0)*onp.random.randn(FLAGS.batch_size, FLAGS.map_size, FLAGS.map_size, 2))
  kappa_init, _ = jax.vmap(ks93)(gamma_init[...,0], gamma_init[...,1])
  # we only care about kappa_e

  samples, trace = tempered_HMC(init_image=kappa_init,
                                total_score_fn=total_score_fn,
                                batch_size=FLAGS.batch_size,
                                initial_temperature=FLAGS.initial_temperature,
                                initial_step_size=FLAGS.initial_step_size,
                                min_steps_per_temp=FLAGS.min_steps_per_temp,
                                num_results=10,
                                num_burnin_steps=0
                                )


  print('average acceptance rate', onp.mean(trace[0]))
  print('final max temperature', onp.max(trace[1][:,-1]))
  # TODO: apply final projection
  # Save the chain
  fits.writeto("./results/"+FLAGS.output_folder+"/samples_"+FLAGS.output_file+".fits", onp.array(samples),overwrite=False)

if __name__ == "__main__":
  app.run(main)
