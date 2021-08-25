# Script for training a denoiser
import os

os.environ['XLA_FLAGS']='--xla_gpu_cuda_data_dir=/gpfslocalsys/cuda/11.1.0'

# Script for sampling constrained realisations
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
import tensorflow as tf

import tensorflow_probability as tfp; tfp = tfp.substrates.jax
from jax_lensing.samplers.score_samplers import ScoreHamiltonianMonteCarlo

from jax_lensing.models.convdae import UResNet18
from jax_lensing.spectral import make_power_map
from jax_lensing.inversion import ks93inv, ks93
from jax_lensing.cluster import gen_nfw_shear

flags.DEFINE_string("output_folder", "100_100_.5_3e14_1", "Name of the output folder.")
flags.DEFINE_string("output_file", "samples.fits", "Filename of output samples.")
#flags.DEFINE_string("shear", "gamma.fits", "Path to input shear maps.")
flags.DEFINE_string("convergence", "convergence.npy", "Path to input noiseless convergence maps.")
flags.DEFINE_string("mask", "mask.fits", "Path to input mask.")
flags.DEFINE_float("sigma_gamma", 0.15, "Standard deviation of shear.")
flags.DEFINE_string("model_weights", "/gpfswork/rech/xdy/commun/Remy2021/score_sn1.0_std0.2/model-40000.pckl", "Path to trained model weights.")
flags.DEFINE_integer("batch_size", 5, "Size of batch to sample in parallel.")
flags.DEFINE_float("initial_temperature", 0.15, "Initial temperature at which to start sampling.")
flags.DEFINE_float("initial_step_size", 0.01, "Initial step size at which to perform sampling.")
flags.DEFINE_integer("min_steps_per_temp", 10, "Minimum number of steps for each temperature.")
flags.DEFINE_integer("num_steps", 5000, "Total number of steps in the chains.")
flags.DEFINE_integer("output_steps", 3, "How many steps to output.")
flags.DEFINE_string("gaussian_path", "data/massivenu/mnu0.0_Maps10_PS_theory.npy", "Path to Massive Nu power spectrum.")
flags.DEFINE_string("std1", "../data/COSMOS/std1.fits", "Standard deviation noise e1 (gal).")
flags.DEFINE_string("std2", "../data/COSMOS/std2.fits", "Standard deviation noise e2 (gal).")
flags.DEFINE_string("cosmos_noise_e1", "../data/COSMOS/cosmos_noise_real1.fits", "Cosmos noise realisation e1.")
flags.DEFINE_string("cosmos_noise_e2", "../data/COSMOS/cosmos_noise_real2.fits", "Cosmos noise realisation e2.")
flags.DEFINE_boolean("cosmos_noise_realisation", False, "Uses Cosmos noise realisation or not.")
flags.DEFINE_boolean("gaussian_only", False, "Only use Gaussian score if yes.")
flags.DEFINE_boolean("reduced_shear", False, "Apply reduced shear correction if yes.")
flags.DEFINE_boolean("gaussian_prior", True, "Uses a Gaussian prior or not.")
flags.DEFINE_float("resolution", 0.29, "Map resoultion arcmin/pixel")
flags.DEFINE_boolean("no_cluster", True, "No added cluster if True")
flags.DEFINE_integer("x_cluster", 100, "x-coordinate of the cluster")
flags.DEFINE_integer("y_cluster", 100, "y-coordinate of the cluster")
flags.DEFINE_float("z_halo", .5, "redshift of the cluster")
flags.DEFINE_float("mass_halo", 2e15, "mass of the cluster (in solar mass)")
flags.DEFINE_float("zs", 1, "redshif of the source")
flags.DEFINE_boolean("COSMOS", False, "COSMOS catalog")
flags.DEFINE_boolean("hmc", False, "Run HMC at high temp before SDE sampling")

FLAGS = flags.FLAGS

def forward_fn(x, s, is_training=False):
  denoiser = UResNet18(n_output_channels=1)
  return denoiser(x, s, is_training=is_training)

def log_gaussian_prior(map_data, sigma, ps_map):
  """ Gaussian prior on the power spectrum of the map
  """
  data_ft = jnp.fft.fft2(map_data) / map_data.shape[0]
  return -0.5*jnp.sum(jnp.real(data_ft*jnp.conj(data_ft)) / (ps_map+sigma**2))
gaussian_prior_score = jax.vmap(jax.grad(log_gaussian_prior), in_axes=[0,0, None])

def log_gaussian_prior_b(map_data, sigma):
    data_ft = jnp.fft.fft2(map_data) / float(360)
    return -0.5*jnp.sum(jnp.real(data_ft*jnp.conj(data_ft)) / (sigma[0]**2))

gaussian_prior_score_b = jax.vmap(jax.grad(log_gaussian_prior_b), in_axes=[0,0])

def main(_):

  std1 = jnp.expand_dims(fits.getdata(FLAGS.std1).astype('float32'), -1)
  std2 = jnp.expand_dims(fits.getdata(FLAGS.std2).astype('float32'), -1)
  sigma_gamma = jnp.concatenate([std1, std2], axis=-1)

  def log_likelihood(x, sigma, meas_shear, sigma_mask):
    """ Likelihood function at the level of the measured shear
    """
    x = x.reshape((360, 360,2))
    ke = x[...,0]
    kb = x[...,1]
    model_shear = jnp.stack(ks93inv(ke, kb), axis=-1)
  
    return - jnp.sum((model_shear - meas_shear)**2/((sigma_gamma)**2 + sigma**2 + sigma_mask) )/2.

  likelihood_score = jax.vmap(jax.grad(log_likelihood), in_axes=[0,0, None, None])

  map_size = fits.getdata(FLAGS.mask).astype('float32').shape[0]

  # Make the network
  #model = hk.transform_with_state(forward_fn)
  model = hk.without_apply_rng(hk.transform_with_state(forward_fn))

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
  kell = ell /2/jnp.pi * 360 * pixel_size / map_size
  # Interpolate the Power Spectrum in Fourier Space
  power_map = jnp.array(make_power_map(ps_halofit, map_size, kps=kell))


  # Load the noiseless convergence map
  if not FLAGS.COSMOS:
    print('i am here')
    convergence= fits.getdata(FLAGS.convergence).astype('float32')
    #convergence = onp.load(FLAGS.convergence).astype('float32') #Convergence is expected in the format [map_size,map_size,1]
  
    # Get the correspinding shear
    gamma1, gamma2 = ks93inv(convergence, onp.zeros_like(convergence)) 
  
    if not FLAGS.no_cluster: 
      print('adding a cluster')
      # Compute NFW profile shear map
      g1_NFW, g2_NFW = gen_nfw_shear(x_cen=FLAGS.x_cluster, y_cen=FLAGS.y_cluster, 
                                   resolution=FLAGS.resolution,
                                   nx=map_size, ny=map_size, z=FLAGS.z_halo,
                                   m=FLAGS.mass_halo, zs=FLAGS.zs)
      # Shear with added NFW cluster
      gamma1 += g1_NFW
      gamma2 += g2_NFW
  

      # Target convergence map with the added cluster
      #ke_cluster, kb_cluster = ks93(g1_cluster, g2_cluster)

    # Add noise the shear map
    if FLAGS.cosmos_noise_realisation:
      print('cosmos noise real')
      gamma1 += fits.getdata(FLAGS.cosmos_noise_e1).astype('float32')
      gamma2 += fits.getdata(FLAGS.cosmos_noise_e2).astype('float32')

    else:
      gamma1 += std1[...,0] * jax.random.normal(jax.random.PRNGKey(42), gamma1.shape) #onp.random.randn(map_size,map_size)
      gamma2 += std2[...,0] * jax.random.normal(jax.random.PRNGKey(43), gamma2.shape) #onp.random.randn(map_size,map_size)

    # Load the shear maps and corresponding mask
    gamma = onp.stack([gamma1, gamma2], -1) # Shear is expected in the format [map_size,map_size,2]

  else:

    # Load the shear maps and corresponding mask
    g1 = fits.getdata('../data/COSMOS/cosmos_full_e1_0.29arcmin360.fits').astype('float32').reshape([map_size, map_size, 1])
    g2 = fits.getdata('../data/COSMOS/cosmos_full_e2_0.29arcmin360.fits').astype('float32').reshape([map_size, map_size, 1])
    gamma = onp.concatenate([g1, g2], axis=-1)

  mask = jnp.expand_dims(fits.getdata(FLAGS.mask).astype('float32'), -1) # has shape [map_size,map_size,1]

  masked_true_shear = gamma * mask
  sigma_mask = (1-mask)*10**3

  #@jax.jit
  #def total_score_fn(x, sigma):
  #  """ Compute the total score, combining the following components:
  #      gaussian prior, ml prior, data likelihood
  #  """
  #  x = x.reshape([FLAGS.batch_size, map_size, map_size])
  #  # Retrieve Gaussian score
  #  gaussian_score = gaussian_prior_score(x, sigma, power_map)
  #  if FLAGS.gaussian_only:
  #    ml_score = 0
  #  else:
  #    # Use Neural network to compute residual prior score
  #    net_input = jnp.stack([x, sigma.reshape([-1,1,1])**2 * gaussian_score], axis=-1)
  #    ml_score = residual_prior_score(net_input, sigma.reshape([-1,1,1,1]))[0][...,0]
  #  # Compute likelihood score
  #  data_score = likelihood_score(x, sigma, gamma, mask)
  #  return (data_score + gaussian_score + ml_score).reshape((-1,map_size*map_size))

  #@jax.jit
  def score_fn(params, state, x, sigma, is_training=False):
    x = x.reshape((-1,360,360,2))
    ke = x[...,0]
    kb = x[...,1]
    
    if FLAGS.gaussian_prior:
      gsb = gaussian_prior_score_b(kb, sigma.reshape((-1,1,1)))
      gsb = jnp.expand_dims(gsb, axis=-1)

      # If requested, first compute the Gaussian prior
      gs = gaussian_prior_score(ke, sigma.reshape((-1,1,1)), power_map)
      gs = jnp.expand_dims(gs, axis=-1)
      #print((jnp.abs(sigma.reshape((-1,1,1,1)))**2).shape, (gs).shape)
      net_input = jnp.concatenate([ke.reshape((-1,360,360,1)), jnp.abs(sigma.reshape((-1,1,1,1)))**2 * gs],axis=-1)
      res, state = model.apply(params, state, net_input, sigma.reshape((-1,1,1,1)), is_training=is_training)
    else:
      res, state = model.apply(params, state, ke.reshape((-1,360,360,1)), sigma.reshape((-1,1,1,1)), is_training=is_training)
      gs = jnp.zeros_like(res)
      gsb = jnp.zeros_like(res)
    return _, res, gs, gsb

  score_fn = partial(score_fn, params, state)
  
  def score_prior(x, sigma):
    #net_input = {'y':x.reshape(-1,360, 360,1), 's':sigma.reshape(-1,1,1,1)}
    _, res, gaussian_score, gsb = score_fn(x.reshape(-1,360, 360,2), sigma.reshape(-1,1,1,1))
    ke = (res[..., 0:1] + gaussian_score).reshape(-1, 360*360)
    kb = gsb[...,0].reshape(-1, 360*360)
    return jnp.stack([ke, kb],axis=-1)
    #return res[..., 0:1].reshape(-1, 360*360), gaussian_score.reshape(-1, 360*360)
  
  @jax.jit
  def total_score_fn(x, sigma):
    sl = likelihood_score(x, sigma, masked_true_shear, sigma_mask).reshape(-1, 360*360,2)
    sp = score_prior(x, sigma)
    return (sl + sp).reshape(-1, 360*360*2)
    #return (sp).reshape(-1, 360*360,2)

  # Prepare the input with a high noise level map

  init_image = jnp.stack([FLAGS.initial_temperature*onp.random.randn(FLAGS.batch_size,360*360),
                         FLAGS.initial_temperature*onp.random.randn(FLAGS.batch_size,360*360)], axis=-1)



  tot_score = partial(total_score_fn, sigma=FLAGS.initial_temperature*jnp.ones((FLAGS.batch_size,1)))

  hmc = ScoreHamiltonianMonteCarlo(
        target_log_prob_fn=None,
        target_score_fn=tot_score,
        step_size=0.01,
        #step_size=10*(np.max(sigma)/s0)**0.5,
        num_leapfrog_steps=3,
        num_delta_logp_steps=4)

  num_results = int(3000)
  num_burnin_steps = int(0)

  def run_chain():
    # Run the chain (with burn-in).
    samples = tfp.mcmc.sample_chain(
        #num_results=num_results
        num_results=2,
        num_steps_between_results=num_results//2,
        num_burnin_steps=num_burnin_steps,
        current_state=init_image.reshape([FLAGS.batch_size,-1]),
        kernel=hmc,
        trace_fn=None,
        #trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
        seed=jax.random.PRNGKey(42))

    #is_accepted = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32))
    return samples#, is_accepted

  if FLAGS.hmc:
    samples = run_chain()
    init_image = samples[-1,...]

  # Run the deterministic chain with the black-box ODE solver
  from scipy import integrate

  @jax.jit
  def dynamics(t, x):
    x = x.reshape([-1,360,360,2])
    return - 0.5*total_score_fn(x, sigma=jnp.ones((FLAGS.batch_size,1,1,1))*jnp.sqrt(t)).reshape([-1])

  noise = FLAGS.initial_temperature

  start_and_end_times = jnp.logspace(jnp.log10(0.99*noise**2),-5, num=50)

  solution = integrate.solve_ivp(dynamics, 
                                [noise**2,(0.0)], 
                                init_image.flatten(),
                                t_eval=start_and_end_times)

   
  # Save the last sample of chain, i.e. x(0) \sim p_0
  samples = solution.y[:,-1].reshape([FLAGS.batch_size,360,360,2])[...,0]

  fits.writeto("./results/"+FLAGS.output_folder+"/samples_"+FLAGS.output_file+".fits", onp.array(samples), overwrite=False)
   
  print('end of sampling')
  # print('average acceptance rate', onp.mean(trace[0]))
  # print('final max temperature', onp.max(trace[1][:,-1]))
  # # TODO: apply final projection
  # # Save the chain
  # fits.writeto("./results/"+FLAGS.output_folder+"/samples_"+FLAGS.output_file+".fits", onp.array(samples),overwrite=False)

if __name__ == "__main__":
  app.run(main)
