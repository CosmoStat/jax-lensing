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
import time


# Import tensorflow for dataset creation and manipulation
import tensorflow as tf

import tensorflow_probability as tfp; tfp = tfp.substrates.jax
from jax_lensing.samplers.score_samplers import ScoreHamiltonianMonteCarlo, ScoreMetropolisAdjustedLangevinAlgorithm
from jax_lensing.samplers.tempered_sampling import TemperedMC


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
flags.DEFINE_integer("x_cluster", 65, "x-coordinate of the cluster")
flags.DEFINE_integer("y_cluster", 130, "y-coordinate of the cluster")
flags.DEFINE_float("z_halo", .5, "redshift of the cluster")
flags.DEFINE_float("mass_halo", 3e14, "mass of the cluster (in solar mass)")
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
  b_mode = False

  std1 = jnp.expand_dims(fits.getdata(FLAGS.std1).astype('float32'), -1)
  std2 = jnp.expand_dims(fits.getdata(FLAGS.std2).astype('float32'), -1)
  sigma_gamma = jnp.concatenate([std1, std2], axis=-1)
  #fits.writeto("./sigma_gamma.fits", onp.array(sigma_gamma), overwrite=False)
  def log_likelihood(x, sigma, meas_shear, mask, sigma_mask):
    """ Likelihood function at the level of the measured shear
    """
    if b_mode:
        x = x.reshape((360, 360,2))
        ke = x[...,0]
        kb = x[...,1]
    else:
        ke = x.reshape((360, 360))
        kb = jnp.zeros(ke.shape)
        
    model_shear = jnp.stack(ks93inv(ke, kb), axis=-1)
    
    return - jnp.sum((model_shear - meas_shear)**2/((sigma_gamma)**2 + sigma**2 + sigma_mask) )/2.

  likelihood_score = jax.vmap(jax.grad(log_likelihood), in_axes=[0,0, None, None, None])

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
  #fits.writeto("./input_shear.fits", onp.array(masked_true_shear), overwrite=False) 

  sigma_mask = (1-mask)*1e10

  
  def score_fn(params, state, x, sigma, is_training=False):
    if b_mode:
        x = x.reshape((-1,360,360,2))
        ke = x[...,0]
        kb = x[...,1]
    else:
        ke = x.reshape((-1,360,360))
    
    if FLAGS.gaussian_prior:
        # If requested, first compute the Gaussian prior
        gs = gaussian_prior_score(ke, sigma.reshape((-1,1,1)), power_map)
        gs = jnp.expand_dims(gs, axis=-1)
        #print((jnp.abs(sigma.reshape((-1,1,1,1)))**2).shape, (gs).shape)
        net_input = jnp.concatenate([ke.reshape((-1,360,360,1)), jnp.abs(sigma.reshape((-1,1,1,1)))**2 * gs],axis=-1)
        res, state = model.apply(params, state, net_input, sigma.reshape((-1,1,1,1)), is_training=is_training)
        if b_mode:
            gsb = gaussian_prior_score_b(kb, sigma.reshape((-1,1,1)))
            gsb = jnp.expand_dims(gsb, axis=-1)
        else:
            gsb = jnp.zeros_like(res)
    else:
        res, state = model.apply(params, state, ke.reshape((-1,360,360,1)), sigma.reshape((-1,1,1,1)), is_training=is_training)
        gs = jnp.zeros_like(res)
        gsb = jnp.zeros_like(res)
    return _, res, gs, gsb

  score_fn = partial(score_fn, params, state)

  def score_prior(x, sigma):
    if b_mode:
        _, res, gaussian_score, gsb = score_fn(x.reshape(-1,360, 360,2), sigma.reshape(-1,1,1,1))
    else:
        _, res, gaussian_score, gsb = score_fn(x.reshape(-1,360, 360), sigma.reshape(-1,1,1))
    ke = (res[..., 0:1] + gaussian_score).reshape(-1, 360*360)
    kb = gsb[...,0].reshape(-1, 360*360)
    if b_mode:
        return jnp.stack([ke, kb],axis=-1)
    else:
        return ke

  def total_score_fn(x, sigma):
    if b_mode:
        sl = likelihood_score(x, sigma, masked_true_shear, mask, sigma_mask).reshape(-1, 360*360,2)
    else:
        sl = likelihood_score(x, sigma, masked_true_shear, mask, sigma_mask).reshape(-1, 360*360)
    sp = score_prior(x, sigma)
    if b_mode:
        return (sl + sp).reshape(-1, 360*360*2)
    else:
        return (sl + sp).reshape(-1, 360*360)
    #return (sp).reshape(-1, 360*360,2)


  # Prepare the input with a high noise level map

  initial_temperature = FLAGS.initial_temperature
  delta_tmp = initial_temperature #onp.sqrt(initial_temperature**2 - 0.148**2)
  initial_step_size = FLAGS.initial_step_size  #0.018
  min_steps_per_temp = FLAGS.min_steps_per_temp #10
  init_image, _ = ks93(mask[...,0]*masked_true_shear[...,0], mask[...,0]*masked_true_shear[...,1])
  init_image = jnp.expand_dims(init_image, axis=0)
  init_image = jnp.repeat(init_image, FLAGS.batch_size, axis=0)
  init_image += (delta_tmp*onp.random.randn(FLAGS.batch_size,360,360))


  def make_kernel_fn(target_log_prob_fn, target_score_fn, sigma):
    return ScoreHamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        target_score_fn=target_score_fn,
        step_size=initial_step_size*(jnp.max(sigma)/initial_temperature)**0.5,
        num_leapfrog_steps=3,
        num_delta_logp_steps=4)

  tmc = TemperedMC(
            target_score_fn=total_score_fn,#score_prior,
            inverse_temperatures=initial_temperature*jnp.ones([FLAGS.batch_size]),
            make_kernel_fn=make_kernel_fn,
            gamma=0.98,
            min_temp=8e-3,
            min_steps_per_temp=min_steps_per_temp,
            num_delta_logp_steps=4)


  num_burnin_steps = int(0)


  samples, trace = tfp.mcmc.sample_chain(
        num_results=2, #FLAGS.num_steps,
        current_state=init_image.reshape([FLAGS.batch_size, -1]),
        kernel=tmc,
        num_burnin_steps=num_burnin_steps,
        num_steps_between_results=6000, #num_results//FLAGS.num_steps,
        trace_fn=lambda _, pkr: (pkr.pre_tempering_results.is_accepted,
                                 pkr.post_tempering_inverse_temperatures,
                                 pkr.tempering_log_accept_ratio),
        seed=jax.random.PRNGKey(int(time.time())))
  
  sol = samples[-1, ...].reshape(-1, 360, 360)

  from scipy import integrate

  @jax.jit
  def dynamics(t, x):
    if b_mode:
      x = x.reshape([-1,360,360,2])
      return - 0.5*total_score_fn(x, sigma=jnp.ones((FLAGS.batch_size,1,1,1))*jnp.sqrt(t)).reshape([-1])
    else:
      x = x.reshape([-1,360,360])
      return - 0.5*total_score_fn(x, sigma=jnp.ones((FLAGS.batch_size,1,1))*jnp.sqrt(t)).reshape([-1])

  init_ode = sol

  last_trace = jnp.mean(trace[1][-1])
  noise = last_trace
  start_and_end_times = jnp.logspace(jnp.log10(0.99*noise**2),-5, num=50)

  solution = integrate.solve_ivp(dynamics, 
                               [noise**2,(1e-5)], 
                               init_ode.flatten(),
                               t_eval=start_and_end_times)

  denoised = solution.y[:,-1].reshape([FLAGS.batch_size,360,360])

  fits.writeto("./results/"+FLAGS.output_folder+"/samples_hmc_"+FLAGS.output_file+".fits", onp.array(sol), overwrite=False)
  fits.writeto("./results/"+FLAGS.output_folder+"/samples_denoised_"+FLAGS.output_file+".fits", onp.array(denoised), overwrite=False)
   
  print('end of sampling')
  # print('average acceptance rate', onp.mean(trace[0]))
  # print('final max temperature', onp.max(trace[1][:,-1]))
  # # TODO: apply final projection
  # # Save the chain
  # fits.writeto("./results/"+FLAGS.output_folder+"/samples_"+FLAGS.output_file+".fits", onp.array(samples),overwrite=False)

if __name__ == "__main__":
  app.run(main)
