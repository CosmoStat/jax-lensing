import tensorflow_datasets.public_api as tfds
import tensorflow as tf
from astropy.io import fits
import numpy as np

# TODO(massivenu): BibTeX citation
_CITATION = """
"""

# TODO(massivenu):
_DESCRIPTION = """
"""

class kappaTNG(tfds.core.GeneratorBasedBuilder):
  """TODO(kappa_tng): Short description of my dataset."""

  # TODO(massive_nu): Set up version.
  VERSION = tfds.core.Version('0.1.0')

  def _info(self):
    # TODO(massive_nu): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
    builder=self,
    # This is the description that will appear on the datasets page.
    description=_DESCRIPTION,
    # tfds.features.FeatureConnectors
    features=tfds.features.FeaturesDict({
       "map": tfds.features.Tensor(shape=[1024,1024], dtype=tf.float32),
    }),
    # If there's a common (input, target) tuple from the features,
    # specify them here. They'll be used if as_supervised=True in
    # builder.as_dataset.
    supervised_keys=("map", "map"),
    homepage='https://dataset-homepage/',
    citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    data_path = dl_manager.extract("/gpfswork/rech/xdy/commun/kappaTNG/kappaTNGCosmos.tar.gz")

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "images_dir_path": os.path.join(data_path, "Maps10"),
            },
        ),
    ]

  def _generate_examples(self, images_dir_path):
    """Yields examples."""
    # TODO(massive_nu): Yields (key, example) tuples from the dataset
    for i, image_file in enumerate(tf.io.gfile.listdir(images_dir_path)):
        with tf.io.gfile.GFile( os.path.join(images_dir_path ,image_file), mode='rb') as f:
          im = 1*fits.getdata(f).astype('float32')
          f.close()

        yield '%d'%i, {"map": im}

