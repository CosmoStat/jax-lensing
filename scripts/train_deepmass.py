# Script for training a DeepMass inference model
import os

os.environ['XLA_FLAGS']='--xla_gpu_cuda_data_dir=/gpfslocalsys/cuda/10.1.2'

from absl import app
from absl import flags

import haiku as hk
import jax
import jax.numpy as jnp
from jax.experimental import optix

import numpy as onp
import pickle
from functools import partial

from flax.metrics import tensorboard

# Import tensorflow for dataset creation and manipulation
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_datasets as tfds

from jax_lensing.models.convdae2 import SmallUResNet, MediumUResNet
from jax_lensing.models.normalization import SNParamsTree as CustomSNParamsTree
from jax_lensing.spectral import make_power_map
from jax_lensing.utils import load_dataset_deepmass
from jax_lensing.inversion import ks93, ks93inv

flags.DEFINE_string("dataset", "kappatng", "Suite of simulations to learn from")
flags.DEFINE_string("output_dir", "./weights/deepmass-sn1v2", "Folder where to store model.")
flags.DEFINE_integer("batch_size", 32, "Size of the batch to train on.")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 45000, "Number of training steps to run.")
flags.DEFINE_string("train_split", "90%", "How much of the training set to use.")
flags.DEFINE_string("mask", "mask.fits", "Path to input mask.")
flags.DEFINE_float("sigma_gamma", 0.15, "Standard deviation of shear.")
flags.DEFINE_float("spectral_norm", 1, "Amount of spectral normalization.")
flags.DEFINE_string("variant", "EiffL", "Variant of model.")
flags.DEFINE_string("model", "MediumUResNet", "Name of model.")
flags.DEFINE_integer("map_size", 360, "Size of maps after cropping")
flags.DEFINE_float("resolution", 0.29, "Resolution in arcmin/pixel")

FLAGS = flags.FLAGS

def forward_fn(x, is_training=False):
  if FLAGS.model == 'SmallUResNet':
    model = SmallUResNet(n_output_channels=1, variant=FLAGS.variant)
  elif FLAGS.model == 'MediumUResNet':
    model = MediumUResNet()
  else:
    raise NotImplementedError
  return model(x, jnp.zeros((1, 1, 1, 1)), is_training=is_training)

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
    sn_fn = hk.transform_with_state(lambda x: CustomSNParamsTree(ignore_regex='[^?!.]*b$',
                                                              val=FLAGS.spectral_norm)(x))

  # Initialisation
  optimizer = optix.chain(
      optix.adam(learning_rate=FLAGS.learning_rate),
      optix.scale_by_schedule(lr_schedule)
  )

  rng_seq = hk.PRNGSequence(42)

  params, state = model.init(next(rng_seq),
                             jnp.zeros((1, FLAGS.map_size, FLAGS.map_size, 1)),
                             is_training=True)

  opt_state = optimizer.init(params)

  if FLAGS.spectral_norm > 0:
    _, sn_state = sn_fn.init(next(rng_seq), params)
  else:
    sn_state = 0.

  mask = jnp.expand_dims(fits.getdata(FLAGS.mask).astype('float32'), 0) # has shape [1, FLAGS.map_size,FLAGS.map_size]

  # Training loss
  def loss_fn(params, state, rng_key, batch):
    key1, key2, key3 = jax.random.split(rng_key, 3)
    # Preprocess the batch for deep mass, i.e. apply KS, add noise, mask, and
    # do inverse Kaiser-Squires
    input_map = batch['x'][..., 0]  # We don't care for the last dimension
    g1, g2 = ks93inv(input_map, jnp.zeros_like(input_map))
    # Add Gaussian noise and mask
    g1 = mask * (g1 + FLAGS.sigma_gamma*jnp.random.normal(key1, g1.shape))
    g2 = mask * (g2 + FLAGS.sigma_gamma*jnp.random.normal(key2, g2.shape))
    ks_map, _ = ks93(g1, g2)
    # Apply model
    res, state = model.apply(params, state, key3, ks_map, is_training=True)
    loss = jnp.mean((input_map - res[..., 0])**2)
    return loss, state

  @jax.jit
  def update(params, state, sn_state, rng_key, opt_state, batch):
    (loss, state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, state, rng_key, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optix.apply_updates(params, updates)
    if FLAGS.spectral_norm > 0:
      new_params, new_sn_state = sn_fn.apply(None, sn_state, None, new_params)
    else:
      new_sn_state = sn_state
    return loss, new_params, state, new_sn_state, new_opt_state

  train = load_dataset_deepmass(FLAGS.dataset, FLAGS.batch_size, FLAGS.map_size, FLAGS.train_split)

  summary_writer = tensorboard.SummaryWriter(FLAGS.output_dir)

  print('training begins')
  for step in range(FLAGS.training_steps):
    loss, params, state, sn_state, opt_state = update(params, state, sn_state,
                                                      next(rng_seq), opt_state,
                                                      next(train))

    summary_writer.scalar('train_loss', loss, step)

    if step%100==0:
        print(step, loss)

    # if step%500==0:
    #   # Running denoiser on a batch of images
    #   batch, res, gs = score_fn(params, state, next(rng_seq), next(train), is_training=False)
    #   summary_writer.image('score/target', onp.clip(batch['x'][0], 0, 0.1)*10., step)
    #   summary_writer.image('score/input', onp.clip(batch['y'][0], 0, 0.1)*10., step)
    #   summary_writer.image('score/score', res[0]+gs[0], step)
    #   summary_writer.image('score/denoised', onp.clip(batch['y'][0] + batch['s'][0,:,:,0]**2 * (res[0]+gs[0]), 0, 0.1)*10., step)
    #   summary_writer.image('score/gaussian_denoised', onp.clip(batch['y'][0] + batch['s'][0,:,:,0]**2 * gs[0], 0, 0.1)*10., step)
    #   print(step)

    if step%5000 ==0:
      with open(FLAGS.output_dir+'/model-%d.pckl'%step, 'wb') as file:
        pickle.dump([params, state, sn_state], file)

  summary_writer.flush()

  with open(FLAGS.output_dir+'/model-final.pckl', 'wb') as file:
    pickle.dump([params, state, sn_state], file)

if __name__ == "__main__":
  app.run(main)
