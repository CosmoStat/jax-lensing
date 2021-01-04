import tensorflow as tf
import tensorflow_datasets as tfds

import jax.numpy as jnp

def load_dataset(batch_size, noise_dist_std, train_split):
  # for training
  def pre_process(im):
    """ Pre-processing function preparing data for denoising task
    """
    # Cutout a portion of the map
    x = tf.image.random_crop(tf.expand_dims(im['map'],-1), [320,320,1])
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    # Sample random Gaussian noise
    u = tf.random.normal(tf.shape(x))
    # Sample standard deviation of noise corruption
    s = noise_dist_std * tf.random.normal((1, 1, 1))
    # Create noisy image
    y = x + s * u
    return {'x':x, 'y':y, 'u':u,'s':s}
  ds = tfds.load('massive_nu', split='train[:{}]'.format(train_split), shuffle_files=True)
  ds = ds.shuffle(buffer_size=10*batch_size)
  ds = ds.repeat()
  ds = ds.map(pre_process)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  return iter(tfds.as_numpy(ds))
