import jax
import jax.image as ji
import jax.numpy as jnp
import haiku as hk

import types
from typing import Mapping, Optional, Sequence, Union

def check_length(length, value, name):
  if len(value) != length:
    raise ValueError(f"`{name}` must be of length 4 not {len(value)}")

def upsample(image):
    b, h, w, c = image.shape
    upsampled_image = ji.resize(image, [b, 2*h, 2*w, c], 'nearest')
    return upsampled_image

class BlockV1NoStride(hk.Module):
  """ResNet V1 block with optional bottleneck."""

  def __init__(
      self,
      channels: int,
      stride: Union[int, Sequence[int]],
      use_projection: bool,
      bn_config: Mapping[str, float],
      bottleneck: bool,
      use_bn: bool = True,
      transpose: bool = False,
      name: Optional[str] = None
  ):
    super().__init__(name=name)
    self.use_projection = use_projection
    self.use_bn = use_bn
    if self.use_bn:
        bn_config = dict(bn_config)
        bn_config.setdefault("create_scale", True)
        bn_config.setdefault("create_offset", True)
        bn_config.setdefault("decay_rate", 0.999)

    if transpose:
      self.pooling_or_upsampling = upsample
    else:
      self.pooling_or_upsampling = hk.AvgPool(window_shape=2, strides=2, padding='SAME')
    self.stride = stride

    if self.use_projection:
      # this is just used for the skip connection
      self.proj_conv = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          # depending on whether it's transpose or not this stride must be
          # replaced by upsampling or avg pooling
          stride=1,
          with_bias=not self.use_bn,
          padding="SAME",
          name="shortcut_conv")
      if self.use_bn:
          self.proj_batchnorm = hk.BatchNorm(name="shortcut_batchnorm", **bn_config)

    channel_div = 4 if bottleneck else 1
    conv_0 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=1 if bottleneck else 3,
        stride=1,
        with_bias=not self.use_bn,
        padding="SAME",
        name="conv_0")
    if self.use_bn:
        bn_0 = hk.BatchNorm(name="batchnorm_0", **bn_config)

    conv_1 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=3,
        stride=1,
        with_bias=not self.use_bn,
        padding="SAME",
        name="conv_1")
    if self.use_bn:
        bn_1 = hk.BatchNorm(name="batchnorm_1", **bn_config)
        layers = ((conv_0, bn_0), (conv_1, bn_1))
    else:
        layers = ((conv_0, None), (conv_1, None))

    if bottleneck:
      conv_2 = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=1,
          with_bias=not self.use_bn,
          padding="SAME",
          name="conv_2")
      if self.use_bn:
          bn_2 = hk.BatchNorm(name="batchnorm_2", scale_init=jnp.zeros, **bn_config)
          layers = layers + ((conv_2, bn_2),)
      else:
          layers = layers + ((conv_2, None),)

    self.layers = layers

  def __call__(self, inputs, is_training, test_local_stats):
    out = shortcut = inputs

    if self.use_projection:
      if self.stride > 1:
          shortcut = self.pooling_or_upsampling(shortcut)
      shortcut = self.proj_conv(shortcut)
      if self.use_bn:
          shortcut = self.proj_batchnorm(shortcut, is_training, test_local_stats)

    if self.stride > 1:
        out = self.pooling_or_upsampling(out)
    for i, (conv_i, bn_i) in enumerate(self.layers):
      out = conv_i(out)
      if self.use_bn:
          out = bn_i(out, is_training, test_local_stats)
      if i < len(self.layers) - 1:  # Don't apply relu on last layer
        out = jax.nn.relu(out)

    return jax.nn.relu(out + shortcut)

class BlockGroup(hk.Module):
  """Higher level block for ResNet implementation."""

  def __init__(
      self,
      channels: int,
      num_blocks: int,
      stride: Union[int, Sequence[int]],
      bn_config: Mapping[str, float],
      bottleneck: bool,
      use_projection: bool,
      transpose: bool,
      use_bn: bool = True,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)

    block_cls = BlockV1NoStride

    self.blocks = []
    for i in range(num_blocks):
      self.blocks.append(
          block_cls(channels=channels,
                    stride=(1 if i else stride),
                    use_projection=(i == 0 and use_projection),
                    bottleneck=bottleneck,
                    bn_config=bn_config,
                    transpose=transpose,
                    use_bn=use_bn,
                    name="block_%d" % (i)))

  def __call__(self, inputs, is_training, test_local_stats):
    out = inputs
    for block in self.blocks:
      out = block(out, is_training, test_local_stats)
    return out

def pad_for_pool(x, n_downsampling):
    problematic_dim = jnp.shape(x)[-2]
    k = jnp.floor_divide(problematic_dim, 2 ** n_downsampling)
    if problematic_dim % 2 ** n_downsampling == 0:
        n_pad = 0
    else:
        n_pad = (k + 1) * 2 ** n_downsampling - problematic_dim
    padding = (n_pad//2, n_pad//2)
    paddings = [
        (0, 0),
        (0, 0),  # here in the context of fastMRI there is nothing to worry about because the dim is 640 (128 x 5)
        # even for brain data, it shouldn't be a problem, since it's 640, 512, or 768.
        padding,
        (0, 0),
    ]
    inputs_padded = jnp.pad(x, paddings)
    return inputs_padded, padding

class UResNet(hk.Module):
  """ Implementation of a denoising auto-encoder based on a resnet architecture
  """

  def __init__(self,
               blocks_per_group,
               bn_config,
               bottleneck,
               channels_per_group,
               use_projection,
               strides,
               n_output_channels=1,
               use_bn=True,
               pad_crop=False,
               variant='EiffL',
               name=None):
    """Constructs a Residual UNet model based on a traditional ResNet.
    Args:
      blocks_per_group: A sequence of length 4 that indicates the number of
        blocks created in each group.
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers. By default the
        ``decay_rate`` is ``0.9`` and ``eps`` is ``1e-5``.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults to
        ``False``.
      bottleneck: Whether the block should bottleneck or not. Defaults to
        ``True``.
      channels_per_group: A sequence of length 4 that indicates the number
        of channels used for each block in each group.
      use_projection: A sequence of length 4 that indicates whether each
        residual block should use projection.
      n_output_channels: The number of output channels, for example to change in
        the case of a complex denoising. Defaults to 1.
      use_bn: Whether the network should use batch normalisation. Defaults to
        ``True``.
      pad_crop: Whether to use cropping/padding to make sure the images can be
        downsampled and upsampled correctly. Defaults to ``False``.
      name: Name of the module.
    """
    super().__init__(name=name)
    self.resnet_v2 = False
    self.use_bn = use_bn
    self.pad_crop = pad_crop
    self.n_output_channels = n_output_channels
    bn_config = dict(bn_config or {})
    bn_config.setdefault("decay_rate", 0.9)
    bn_config.setdefault("eps", 1e-5)
    bn_config.setdefault("create_scale", True)
    bn_config.setdefault("create_offset", True)
    self.variant = variant
    self.strides = strides
    bl = len(self.strides)

    # Number of blocks in each group for ResNet.
    check_length(bl, blocks_per_group, "blocks_per_group")
    check_length(bl, channels_per_group, "channels_per_group")
    self.upsampling = upsample
    self.pooling = hk.AvgPool(window_shape=2, strides=2, padding='SAME')

    self.initial_conv = hk.Conv2D(
        output_channels=32,
        kernel_shape=7,
        stride=1,
        with_bias=not self.use_bn,
        padding="SAME",
        name="initial_conv")

    if not self.resnet_v2 and self.use_bn:
      self.initial_batchnorm = hk.BatchNorm(name="initial_batchnorm",
                                            **bn_config)

    self.block_groups = []
    self.up_block_groups = []
    for i in range(bl):
      self.block_groups.append(
          BlockGroup(channels=channels_per_group[i],
                     num_blocks=blocks_per_group[i],
                     stride=strides[i],
                     bn_config=bn_config,
                     bottleneck=bottleneck,
                     use_projection=use_projection[i],
                     transpose=False,
                     use_bn=self.use_bn,
                     name="block_group_%d" % (i)))

    for i in range(bl):
      self.up_block_groups.append(
          BlockGroup(channels=channels_per_group[i],
                     num_blocks=blocks_per_group[i],
                     stride=strides[i],
                     bn_config=bn_config,
                     bottleneck=bottleneck,
                     use_projection=use_projection[i],
                     transpose=True,
                     use_bn=self.use_bn,
                     name="up_block_group_%d" % (i)))

    if self.resnet_v2 and self.use_bn:
      self.final_batchnorm = hk.BatchNorm(name="final_batchnorm", **bn_config)

    if self.variant == "Zacc":
      self.final_up_conv = hk.Conv2D(
          output_channels=channels_per_group[0]*self.n_output_channels//2,
          kernel_shape=5,
          stride=1,
          padding="SAME",
          name="final_up_conv",
      )
      self.antepenultian_conv = hk.Conv2D(
          output_channels=channels_per_group[0]*self.n_output_channels//2,
          kernel_shape=3,
          stride=1,
          padding='SAME',
          name='antepenultian_conv',
      )
    self.final_conv = hk.Conv2D(
        output_channels=self.n_output_channels,
        kernel_shape=5,
        stride=1,
        padding='SAME',
        name='final_conv',
    )


  def __call__(self, inputs, condition, is_training, test_local_stats=False):
    out = inputs
    if self.pad_crop:
        out, padding = pad_for_pool(inputs, 4)
    
    out = jnp.concatenate([out, condition*jnp.ones_like(out)[...,[0]]], axis=-1)
    out = self.initial_conv(out)
    if self.variant == "Zacc":
      out = self.pooling(out)
    # Decreasing resolution
    levels = []
    for block_group in self.block_groups:
      levels.append(out)
      out = block_group(out, is_training, test_local_stats)
    out = jnp.concatenate([out, condition*jnp.ones_like(out)],axis=-1)
    
    # Increasing resolution
    for i, block_group in enumerate(self.up_block_groups[::-1]):
      out = block_group(out, is_training, test_local_stats)
      out = jnp.concatenate([out, levels[-i-1]],axis=-1)

    # Second to last upsampling, merging with input branch
    if self.variant == "Zacc":
      out = self.upsampling(out)
      out = self.final_up_conv(out)
      out = self.antepenultian_conv(out)
      out = jax.nn.relu(out)

    out = self.final_conv(out)
    if self.pad_crop:
        condition_normalisation = (jnp.abs(condition)*jnp.ones_like(pad_for_pool(inputs, 4)[0])+1e-3)
    else:
        condition_normalisation = (jnp.abs(condition)*jnp.ones_like(inputs)+1e-3)
    out = out / condition_normalisation
    if self.pad_crop:
        if not jnp.sum(padding) == 0:
            out = out[:, :, padding[0]:-padding[1]]
    return out

class SmallUResNet(UResNet):
  """ResNet18."""

  def __init__(self,
               bn_config: Optional[Mapping[str, float]] = None,
               use_bn: bool = True,
               pad_crop: bool = False,
               n_output_channels: int = 1,
               variant: Optional[str] = 'EiffL',
               name: Optional[str] = None):
    """Constructs a ResNet model.
    Args:
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to ``False``.
      use_bn: Whether the network should use batch normalisation. Defaults to
        ``True``.
      n_output_channels: The number of output channels, for example to change in
        the case of a complex denoising. Defaults to 1.
      name: Name of the module.
    """
    super().__init__(blocks_per_group=(2, 2, 2, 2),
                     bn_config=bn_config,
                     bottleneck=False,
                     channels_per_group=(32, 64, 128, 128),
                     use_projection=(True, True, True, True),
                     # 320 -> 160 -> 80 -> 40
                     # 360 -> 180 -> 90 -> 45
                     strides=(2, 2, 2, 1),
                     use_bn=use_bn,
                     pad_crop=pad_crop,
                     n_output_channels=n_output_channels,
                     variant=variant,
                     name=name)
