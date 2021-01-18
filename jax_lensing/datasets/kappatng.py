"""kappatng dataset."""

import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# TODO(kappatng): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(kappatng): BibTeX citation
_CITATION = """
"""

class KappaTNGConfig(tfds.core.BuilderConfig):
  """BuilderConfig for KappaTNG."""

  def __init__(self, *, resolution, **kwargs):
    """BuilderConfig for SQUAD.
    Args:
      resolution: Resolution of the image. Values supported: powers of 2 up to
        1024.
      **kwargs: keyword arguments forwarded to super.
    """
    v2 = tfds.core.Version("2.0.0")
    super(KappaTNGConfig, self).__init__(
        name="%d" % resolution,
        description=("KappaTNG images in %d x %d resolution" %
                     (resolution, resolution)),
        version=v2,
        release_notes={
            "2.0.0": "New split API (https://tensorflow.org/datasets/splits)",
        },
        **kwargs)
    self.resolution = resolution
    self.file_name = "kappaTNGCosmos.tar.gz"

class Kappatng(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for kappatng dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  manual_dir should contain a tar file with images (KappaTNGCosmos.tar.gz).
  """
  
  BUILDER_CONFIGS = [KappaTNGConfig(resolution=1024)]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(kappatng): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            "map": tfds.features.Tensor(shape=[1024,1024], dtype=tf.float32),
	}),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=("map", "map"),
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(kappatng): Downloads the data and defines the splits
    data_path = dl_manager.extract(os.path.join(dl_manager.manual_dir, self.builder_config.file_name)) 

    # TODO(kappatng): Returns the Dict[split names, Iterator[Key, Example]]
    return [

        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
            "images_dir_path": os.path.join(data_path, "global/cscratch1/sd/jialiu/kappaTNG/COSMOS/"),
            },
        ),
    ]

  def _generate_examples(self, images_dir_path):
    """Yields examples."""
    # TODO(kappatng): Yields (key, example) tuples from the dataset
    #yield 'key', {}
    for i, image_file in enumerate(os.listdir(images_dir_path)):
      with open(os.path.join(images_dir_path, image_file), mode="rb") as f:
        im = np.load(f).astype("float32")*1
        f.close()
      yield '%d'%i, {"map": im}
