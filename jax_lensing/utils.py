import tensorflow as tf
import tensorflow_datasets as tfds

import jax
import jax.numpy as jnp

def load_dataset(name, batch_size, crop_width, noise_dist_std, train_split):
  # for training
  def pre_process(im):
    """ Pre-processing function preparing data for denoising task
    """
    # Cutout a portion of the map
    x = tf.image.random_crop(tf.expand_dims(im['map'],-1), [crop_width,crop_width,1])
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    # Sample random Gaussian noise
    u = tf.random.normal(tf.shape(x))
    # Sample standard deviation of noise corruption
    s = noise_dist_std * tf.random.normal((1, 1, 1))
    # Create noisy image
    y = x + s * u
    return {'x':x, 'y':y, 'u':u,'s':s}
  ds = tfds.load(name, split='train[:{}]'.format(train_split), shuffle_files=True)
  ds = ds.shuffle(buffer_size=10*batch_size)
  ds = ds.repeat()
  ds = ds.map(pre_process)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  return iter(tfds.as_numpy(ds))

def load_dataset_deepmass(name, batch_size, crop_width, train_split):
  """
  This is a simplified version of the previous function to load datasets, here
  we only want retrieve the maps, the rest of the preprocessing needs some jax
  functions.
  """
  def pre_process(im):
    """ Pre-processing function preparing data for denoising task
    """
    # Cutout a portion of the map
    x = tf.image.random_crop(tf.expand_dims(im['map'],-1), [crop_width,crop_width,1])
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return {'x':x }
  ds = tfds.load(name, split='train[:{}]'.format(train_split), shuffle_files=True)
  ds = ds.shuffle(buffer_size=10*batch_size)
  ds = ds.repeat()
  ds = ds.map(pre_process)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  return iter(tfds.as_numpy(ds))

def bin2d(x, y, npix=10, v=None, w=None, size=None, verbose=False):

  # Compute weighted bin count map 
  wmap, xbins, ybins = jnp.histogram2d(x, y, bins=npix, range=[-size/2, size/2],
                                        weights=w)
  # Handle division by zero (i.e., empty pixels)
  #wmap = jax.ops.index_update(wmap, jax.ops.index[jnp.where(wmap==0)], jnp.inf)
  contours = jnp.load('../data/COSMOS/contours.npy')
  wmap = wmap + contours
  # Compute mean values per pixel
  result = (jnp.histogram2d(x, y, bins=npix, range=[-size/2, size/2],
                    weights=(v * w))[0] / wmap).T

  return result

def random_rotations(e1, e2, n, rng_key):
  gamma1 = jnp.repeat(jnp.expand_dims(e1, 0), n, axis=0)
  gamma2 = jnp.repeat(jnp.expand_dims(e2, 0), n, axis=0)
  theta = jnp.pi * jax.random.normal(rng_key, gamma1.shape)
  new_gamma1 = jnp.cos(theta) * gamma1 - jnp.sin(theta) * gamma2
  new_gamma2 = jnp.sin(theta) * gamma1 + jnp.cos(theta) * gamma2
  return new_gamma1, new_gamma2
