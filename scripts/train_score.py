# Script for training a denoiser
from absl import app
from absl import flags

from flax.metrics import tensorboard
import haiku as hk
import optax

import jax
import jax.numpy as jnp

import numpy as onp
import pickle
from functools import partial

# Import tensorflow for dataset creation and manipulation
import tensorflow as tf
import tensorflow_datasets as tfds

from jax_lensing.models import UResNet18
from jax_lensing.models.normalization import SNParamsTree 
from jax_lensing.spectral import make_power_map
from jax_lensing.utils import load_dataset

flags.DEFINE_string("dataset", "kappatng", "Suite of simulations to learn from")
flags.DEFINE_string("output_dir", "./weights/gp-sn1v5", "Folder where to store model.")
flags.DEFINE_integer("batch_size", 32, "Size of the batch to train on.")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 45000, "Number of training steps to run.")
flags.DEFINE_string("train_split", "90%", "How much of the training set to use.")
flags.DEFINE_float("noise_dist_std", 0.2, "Standard deviation of the noise distribution.")
flags.DEFINE_float("spectral_norm", 1, "Amount of spectral normalization.")
flags.DEFINE_boolean("gaussian_prior", True, "Whether to train including Gaussian prior information.")
flags.DEFINE_string("gaussian_path", "data/ktng/ktng_PS_theory.npy", "Path to Massive Nu power spectrum.")
flags.DEFINE_integer("map_size", 360, "Size of maps after cropping")
flags.DEFINE_float("resolution", 0.29, "Resolution in arcmin/pixel")

FLAGS = flags.FLAGS

def forward_fn(x, s, is_training=False):
  denoiser = UResNet18(n_output_channels=1, pad_crop=True)
  return denoiser(x, s, is_training=is_training)

def log_gaussian_prior(map_data, sigma, ps_map):
  data_ft = jnp.fft.fft2(map_data) / float(FLAGS.map_size)
  return -0.5*jnp.sum(jnp.real(data_ft*jnp.conj(data_ft)) / (ps_map+sigma[0]**2))
gaussian_prior_score = jax.vmap(jax.grad(log_gaussian_prior), in_axes=[0,0, None])

def lr_schedule(step):
  """Linear scaling rule optimized for 90 epochs."""
  steps_per_epoch = 30000 // FLAGS.batch_size

  current_epoch = step / steps_per_epoch  # type: float
  lr = (1.0 * FLAGS.batch_size) / 32
  boundaries = jnp.array((20, 40, 60)) * steps_per_epoch
  values = jnp.array([1., 0.1, 0.01, 0.001]) * lr

  index = jnp.sum(boundaries < step)
  return jnp.take(values, index)

def main(_):
  # Make the network
  model = hk.transform_with_state(forward_fn)

  if FLAGS.spectral_norm > 0:
    sn_fn = hk.transform_with_state(lambda x: SNParamsTree(ignore_regex='[^?!.]*b$|[^?!.]*offset$',
                                                              val=FLAGS.spectral_norm)(x))

  # Initialisation
  optimizer = optax.chain(
      optax.adam(learning_rate=FLAGS.learning_rate),
      optax.scale_by_schedule(lr_schedule)
  )

  rng_seq = hk.PRNGSequence(42)

  if FLAGS.gaussian_prior:
    last_dim=2
  else:
    last_dim=1

  params, state = model.init(next(rng_seq),
                             jnp.zeros((1, FLAGS.map_size, FLAGS.map_size, last_dim)),
                             jnp.zeros((1, 1, 1, 1)), is_training=True)

  opt_state = optimizer.init(params)

  if FLAGS.spectral_norm > 0:
    _, sn_state = sn_fn.init(next(rng_seq), params)
  else:
    sn_state = 0.

  # If the Gaussian prior is used, load the theoretical power spectrum
  pixel_size = jnp.pi * FLAGS.resolution / 180. / 60. #rad/pixel
  if FLAGS.gaussian_prior:
    ps_data = onp.load(FLAGS.gaussian_path).astype('float32')
    ell = jnp.array(ps_data[0,:])
    # massivenu: channel 4
    ps_halofit = jnp.array(ps_data[1,:] / pixel_size**2) # normalisation by pixel size
    # convert to pixel units of our simple power spectrum calculator
    kell = ell /2/jnp.pi * 360 * pixel_size / FLAGS.map_size
    # Interpolate the Power Spectrum in Fourier Space
    power_map = jnp.array(make_power_map(ps_halofit, FLAGS.map_size, kps=kell))

  def score_fn(params, state, rng_key, batch, is_training=True):
    if FLAGS.gaussian_prior:
      # If requested, first compute the Gaussian prior
      gaussian_score = gaussian_prior_score(batch['y'][...,0], batch['s'][...,0], power_map)
      gaussian_score = jnp.expand_dims(gaussian_score, axis=-1)
      net_input = jnp.concatenate([batch['y'], jnp.abs(batch['s'])**2 * gaussian_score],axis=-1)
      res, state = model.apply(params, state, rng_key, net_input, batch['s'], is_training=is_training)
    else:
      res, state = model.apply(params, state, rng_key, batch['y'], batch['s'], is_training=is_training)
      gaussian_score = jnp.zeros_like(res)
    return batch, res, state, gaussian_score

  # Training loss
  def loss_fn(params, state, rng_key, batch):
    _, res, state, gaussian_score = score_fn(params, state, rng_key, batch)
    loss = jnp.mean((batch['u'] + batch['s'] * (res + gaussian_score))**2)
    return loss, state

  @jax.jit
  def update(params, state, sn_state, rng_key, opt_state, batch):
    (loss, state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, state, rng_key, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    if FLAGS.spectral_norm > 0:
      new_params, new_sn_state = sn_fn.apply(None, sn_state, None, new_params)
    else:
      new_sn_state = sn_state
    return loss, new_params, state, new_sn_state, new_opt_state

  train = load_dataset(FLAGS.dataset, FLAGS.batch_size, FLAGS.map_size, FLAGS.noise_dist_std, FLAGS.train_split)

  summary_writer = tensorboard.SummaryWriter(FLAGS.output_dir)

  print('training begins')
  for step in range(FLAGS.training_steps):
    loss, params, state, sn_state, opt_state = update(params, state, sn_state,
                                                      next(rng_seq), opt_state,
                                                      next(train))

    summary_writer.scalar('train_loss', loss, step)

    if step%100==0:
        print(step, loss)

    if step%500==0:
      # Running denoiser on a batch of images
      batch, res, gs = score_fn(params, state, next(rng_seq), next(train), is_training=False)
      summary_writer.image('score/target', onp.clip(batch['x'][0], 0, 0.1)*10., step)
      summary_writer.image('score/input', onp.clip(batch['y'][0], 0, 0.1)*10., step)
      summary_writer.image('score/score', res[0]+gs[0], step)
      summary_writer.image('score/denoised', onp.clip(batch['y'][0] + batch['s'][0,:,:,0]**2 * (res[0]+gs[0]), 0, 0.1)*10., step)
      summary_writer.image('score/gaussian_denoised', onp.clip(batch['y'][0] + batch['s'][0,:,:,0]**2 * gs[0], 0, 0.1)*10., step)
      print(step)

    if step%5000 ==0:
      with open(FLAGS.output_dir+'/model-%d.pckl'%step, 'wb') as file:
        pickle.dump([params, state, sn_state], file)

  summary_writer.flush()

  with open(FLAGS.output_dir+'/model-final.pckl', 'wb') as file:
    pickle.dump([params, state, sn_state], file)

if __name__ == "__main__":
  app.run(main)
