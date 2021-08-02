"""Score-based MCMC samplers"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax
import jax.numpy as jnp
import tensorflow_probability as tfp; tfp = tfp.substrates.jax
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

__all__ = [
    'ScoreUncalibratedHamiltonianMonteCarlo',
    'ScoreUncalibratedLangevin',
    'ScoreHamitonianMonteCarlo',
    'ScoreMetropolisAdjustedLangevinAlgorithm'
]

class ScoreUncalibratedHamiltonianMonteCarlo(tfp.mcmc.UncalibratedHamiltonianMonteCarlo):
  def __init__(self,
               target_score_fn,
               step_size,
               num_leapfrog_steps,
               num_delta_logp_steps,
               target_log_prob_fn=None,
               state_gradients_are_stopped=False,
               store_parameters_in_results=False,
               experimental_shard_axis_names=None,
               name=None):

    if target_log_prob_fn is None:
      # We begin by creating a fake logp, with the correct scores
      @jax.custom_jvp
      def fake_logp(x):
        return 0.
      @fake_logp.defjvp
      def fake_logp_jvp(primals, tangents):
        x, = primals
        x_dot, = tangents
        primal_out = fake_logp(x)
        s = target_score_fn(x)
        tangent_out = x_dot.dot(s)
        return primal_out, tangent_out
      target_log_prob_fn = fake_logp

    super().__init__(target_log_prob_fn,
                     step_size,
                     num_leapfrog_steps,
                     state_gradients_are_stopped=state_gradients_are_stopped,
                     name=name,
                     experimental_shard_axis_names=experimental_shard_axis_names,
                     store_parameters_in_results=store_parameters_in_results)
    self._parameters['target_score_fn'] = target_score_fn
    self._parameters['num_delta_logp_steps'] = num_delta_logp_steps

  def one_step(self, current_state, previous_kernel_results, seed=None):
    """
    Wrapper over the normal HMC steps
    """
    next_state_parts, new_kernel_results = super().one_step(current_state,
                                                            previous_kernel_results,
                                                            seed)
    # We need to integrate the score over a path between input and output points
    # Direction of integration
    if mcmc_util.is_list_like(current_state):
      v = next_state_parts[0] - current_state[0]
      cs = current_state[0]
    else:
      v = next_state_parts - current_state
      cs = current_state
    @jax.vmap
    def integrand(t):
      return jnp.sum(self._parameters['target_score_fn']( t * v + cs)*v, axis=-1)
    delta_logp = simps(integrand,0.,1., self._parameters['num_delta_logp_steps'])
    new_kernel_results2 = new_kernel_results._replace(log_acceptance_correction=new_kernel_results.log_acceptance_correction + delta_logp)
    return next_state_parts, new_kernel_results2

class ScoreUncalibratedLangevin(tfp.mcmc.UncalibratedLangevin):
  def __init__(self,
               target_score_fn,
               step_size,
               num_delta_logp_steps,
               volatility_fn=None,
               target_log_prob_fn=None,
               parallel_iterations=10,
               compute_acceptance=True,
               seed=None,
               name=None):

    if target_log_prob_fn is None:
      # We begin by creating a fake logp, with the correct scores
      @jax.custom_jvp
      def fake_logp(x):
        return 0.
      @fake_logp.defjvp
      def fake_logp_jvp(primals, tangents):
        x, = primals
        x_dot, = tangents
        primal_out = fake_logp(x)
        s = target_score_fn(x)
        tangent_out = x_dot.dot(s)
        return primal_out, tangent_out
      target_log_prob_fn = fake_logp

    super().__init__(target_log_prob_fn,
                     step_size,
                     volatility_fn=volatility_fn,
                     parallel_iterations=parallel_iterations,
                     compute_acceptance=compute_acceptance,
                     seed=seed,
                     name=name)
    self._parameters['target_score_fn'] = target_score_fn
    self._parameters['num_delta_logp_steps'] = num_delta_logp_steps

  def one_step(self, current_state, previous_kernel_results, seed=None):
    """
    Wrapper over the normal Langevin step
    """
    next_state_parts, new_kernel_results = super().one_step(current_state,
                                                            previous_kernel_results,
                                                            seed)
    # We need to integrate the score over a path between input and output points
    # Direction of integration
    v = next_state_parts - current_state
    @jax.vmap
    def integrand(t):
      return self._parameters['target_score_fn']( t * v + current_state).dot(v)
    delta_logp = simps(integrand,0.,1., self._parameters['num_delta_logp_steps'])
    new_kernel_results2 = new_kernel_results._replace(log_acceptance_correction=new_kernel_results.log_acceptance_correction + delta_logp)
    return next_state_parts, new_kernel_results2

class ScoreHamiltonianMonteCarlo(tfp.mcmc.HamiltonianMonteCarlo):

  def __init__(self,
               target_score_fn,
               step_size,
               num_leapfrog_steps,
               num_delta_logp_steps,
               state_gradients_are_stopped=False,
               step_size_update_fn=None,
               target_log_prob_fn=None,
               seed=None,
               store_parameters_in_results=False,
               experimental_shard_axis_names=None,
               name=None):
    """Initializes this transition kernel.
    Args:
      target_score_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns the score
        of the log-density under the target distribution.
      step_size: `Tensor` or Python `list` of `Tensor`s representing the step
        size for the leapfrog integrator. Must broadcast with the shape of
        `current_state`. Larger step sizes lead to faster progress, but
        too-large step sizes make rejection exponentially more likely. When
        possible, it's often helpful to match per-variable step sizes to the
        standard deviations of the target distribution in each variable.
      num_leapfrog_steps: Integer number of steps to run the leapfrog integrator
        for. Total progress per HMC step is roughly proportional to
        `step_size * num_leapfrog_steps`.
      num_delta_logp_steps: Integer number of steps to run the integrator
        for estimating the change in logp.
      state_gradients_are_stopped: Python `bool` indicating that the proposed
        new state be run through `tf.stop_gradient`. This is particularly useful
        when combining optimization over samples from the HMC chain.
        Default value: `False` (i.e., do not apply `stop_gradient`).
      step_size_update_fn: Python `callable` taking current `step_size`
        (typically a `tf.Variable`) and `kernel_results` (typically
        `collections.namedtuple`) and returns updated step_size (`Tensor`s).
        Default value: `None` (i.e., do not update `step_size` automatically).
      seed: Python integer to seed the random number generator. Deprecated, pass
        seed to `tfp.mcmc.sample_chain`.
      store_parameters_in_results: If `True`, then `step_size` and
        `num_leapfrog_steps` are written to and read from eponymous fields in
        the kernel results objects returned from `one_step` and
        `bootstrap_results`. This allows wrapper kernels to adjust those
        parameters on the fly. This is incompatible with `step_size_update_fn`,
        which must be set to `None`.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'hmc_kernel').
    """
    if step_size_update_fn and store_parameters_in_results:
      raise ValueError('It is invalid to simultaneously specify '
                       '`step_size_update_fn` and set '
                       '`store_parameters_in_results` to `True`.')
    self._seed_stream = tfp.util.SeedStream(seed, salt='hmc')
    uhmc_kwargs = {} if seed is None else dict(seed=self._seed_stream())
    mh_kwargs = {} if seed is None else dict(seed=self._seed_stream())
    self._impl = tfp.mcmc.MetropolisHastings(
        inner_kernel=ScoreUncalibratedHamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            target_score_fn=target_score_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            num_delta_logp_steps=num_delta_logp_steps,
            state_gradients_are_stopped=state_gradients_are_stopped,
            name=name or 'hmc_kernel',
            store_parameters_in_results=store_parameters_in_results,
            **uhmc_kwargs),
        **mh_kwargs).experimental_with_shard_axes(experimental_shard_axis_names)
    self._parameters = self._impl.inner_kernel.parameters.copy()
    self._parameters['step_size_update_fn'] = step_size_update_fn
    self._parameters['seed'] = seed


class ScoreMetropolisAdjustedLangevinAlgorithm(tfp.mcmc.MetropolisAdjustedLangevinAlgorithm):
  """Runs one step of Metropolis-adjusted Langevin algorithm.
  Metropolis-adjusted Langevin algorithm (MALA) is a Markov chain Monte Carlo
  (MCMC) algorithm that takes a step of a discretised Langevin diffusion as a
  proposal. This class implements one step of MALA using Euler-Maruyama method
  for a given `current_state` and diagonal preconditioning `volatility` matrix.
  Mathematical details and derivations can be found in
  [Roberts and Rosenthal (1998)][1] and [Xifara et al. (2013)][2].
  See `UncalibratedLangevin` class description below for details on the proposal
  generating step of the algorithm.
  The `one_step` function can update multiple chains in parallel. It assumes
  that all leftmost dimensions of `current_state` index independent chain states
  (and are therefore updated independently). The output of
  `target_log_prob_fn(*current_state)` should reduce log-probabilities across
  all event dimensions. Slices along the rightmost dimensions may have different
  target distributions; for example, `current_state[0, :]` could have a
  different target distribution from `current_state[1, :]`. These semantics are
  governed by `target_log_prob_fn(*current_state)`. (The number of independent
  chains is `tf.size(target_log_prob_fn(*current_state))`.)
  #### Examples:
  ##### Simple chain with warm-up.
  In this example we sample from a standard univariate normal
  distribution using MALA with `step_size` equal to 0.75.
  ```python
  from tensorflow_probability.python.internal.backend.jax.compat import v2 as tf
  import tensorflow_probability as tfp; tfp = tfp.experimental.substrates.jax
  import numpy as np
  import matplotlib.pyplot as plt
  tf.enable_v2_behavior()
  tfd = tfp.distributions
  dtype = np.float32
  # Target distribution is Standard Univariate Normal
  target = tfd.Normal(loc=dtype(0), scale=dtype(1))
  def target_log_prob(x):
    return target.log_prob(x)
  # Define MALA sampler with `step_size` equal to 0.75
  samples = tfp.mcmc.sample_chain(
      num_results=1000,
      current_state=dtype(1),
      kernel=tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
          target_log_prob_fn=target_log_prob,
          step_size=0.75),
      num_burnin_steps=500,
      trace_fn=None,
      seed=42)
  sample_mean = tf.reduce_mean(samples, axis=0)
  sample_std = tf.sqrt(
      tf.reduce_mean(
          tf.math.squared_difference(samples, sample_mean),
          axis=0))
  print('sample mean', sample_mean)
  print('sample standard deviation', sample_std)
  plt.title('Traceplot')
  plt.plot(samples.numpy(), 'b')
  plt.xlabel('Iteration')
  plt.ylabel('Position')
  plt.show()
  ```
  ##### Sample from a 3-D Multivariate Normal distribution.
  In this example we also consider a non-constant volatility function.
  ```python
  from tensorflow_probability.python.internal.backend.jax.compat import v2 as tf
  import tensorflow_probability as tfp; tfp = tfp.experimental.substrates.jax
  import numpy as np
  tf.enable_v2_behavior()
  dtype = np.float32
  true_mean = dtype([0, 0, 0])
  true_cov = dtype([[1, 0.25, 0.25], [0.25, 1, 0.25], [0.25, 0.25, 1]])
  num_results = 500
  num_chains = 500
  # Target distribution is defined through the Cholesky decomposition
  chol = tf.linalg.cholesky(true_cov)
  target = tfd.MultivariateNormalTriL(loc=true_mean, scale_tril=chol)
  # Here we define the volatility function to be non-constant
  def volatility_fn(x):
    # Stack the input tensors together
    return 1. / (0.5 + 0.1 * tf.math.abs(x))
  # Initial state of the chain
  init_state = np.ones([num_chains, 3], dtype=dtype)
  # Run MALA with normal proposal for `num_results` iterations for
  # `num_chains` independent chains:
  states = tfp.mcmc.sample_chain(
      num_results=num_results,
      current_state=init_state,
      kernel=tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
          target_log_prob_fn=target.log_prob,
          step_size=.1,
          volatility_fn=volatility_fn),
      num_burnin_steps=200,
      num_steps_between_results=1,
      trace_fn=None,
      seed=42)
  sample_mean = tf.reduce_mean(states, axis=[0, 1])
  x = (states - sample_mean)[..., tf.newaxis]
  sample_cov = tf.reduce_mean(
      tf.matmul(x, tf.transpose(x, [0, 1, 3, 2])), [0, 1])
  print('sample mean', sample_mean.numpy())
  print('sample covariance matrix', sample_cov.numpy())
  ```
  #### References
  [1]: Gareth Roberts and Jeffrey Rosenthal. Optimal Scaling of Discrete
       Approximations to Langevin Diffusions. _Journal of the Royal Statistical
       Society: Series B (Statistical Methodology)_, 60: 255-268, 1998.
       https://doi.org/10.1111/1467-9868.00123
  [2]: T. Xifara et al. Langevin diffusions and the Metropolis-adjusted
       Langevin algorithm. _arXiv preprint arXiv:1309.2983_, 2013.
       https://arxiv.org/abs/1309.2983
  """
  #
  # @deprecation.deprecated_args(
  #     '2020-09-20', 'The `seed` argument is deprecated (but will work until '
  #     'removed). Pass seed to `tfp.mcmc.sample_chain` instead.', 'seed')
  def __init__(self,
               target_score_fn,
               step_size,
               num_delta_logp_steps,
               volatility_fn=None,
               target_log_prob_fn=None,
               seed=None,
               parallel_iterations=10,
               name=None):
    """Initializes MALA transition kernel.
    Args:
      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
      step_size: `Tensor` or Python `list` of `Tensor`s representing the step
        size for the leapfrog integrator. Must broadcast with the shape of
        `current_state`. Larger step sizes lead to faster progress, but
        too-large step sizes make rejection exponentially more likely. When
        possible, it's often helpful to match per-variable step sizes to the
        standard deviations of the target distribution in each variable.
      num_delta_logp_steps: Integer number of steps to run the integrator
        for estimating the change in logp.
      volatility_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns
        volatility value at `current_state`. Should return a `Tensor` or Python
        `list` of `Tensor`s that must broadcast with the shape of
        `current_state` Defaults to the identity function.
      seed: Python integer to seed the random number generator. Deprecated, pass
        seed to `tfp.mcmc.sample_chain`.
      parallel_iterations: the number of coordinates for which the gradients of
        the volatility matrix `volatility_fn` can be computed in parallel.
        Default value: `None` (i.e., no seed).
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'mala_kernel').
    Returns:
      next_state: Tensor or Python list of `Tensor`s representing the state(s)
        of the Markov chain(s) at each result step. Has same shape as
        `current_state`.
      kernel_results: `collections.namedtuple` of internal calculations used to
        advance the chain.
    Raises:
      ValueError: if there isn't one `step_size` or a list with same length as
        `current_state`.
      TypeError: if `volatility_fn` is not callable.
    """
    seed_stream = tfp.util.SeedStream(seed, salt='langevin')
    mh_kwargs = {} if seed is None else dict(seed=seed_stream())
    uncal_kwargs = {} if seed is None else dict(seed=seed_stream())
    impl = tfp.mcmc.MetropolisHastings(
        inner_kernel=ScoreUncalibratedLangevin(
            target_log_prob_fn=target_log_prob_fn,
            target_score_fn=target_score_fn,
            step_size=step_size,
            num_delta_logp_steps=num_delta_logp_steps,
            volatility_fn=volatility_fn,
            parallel_iterations=parallel_iterations,
            name=name,
            **uncal_kwargs),
        **mh_kwargs)

    self._impl = impl
    parameters = impl.inner_kernel.parameters.copy()
    # Remove `compute_acceptance` parameter as this is not a MALA kernel
    # `__init__` parameter.
    del parameters['compute_acceptance']
    self._parameters = parameters


def simps(f, a, b, N=128):
    """Approximate the integral of f(x) from a to b by Simpson's rule.
    Simpson's rule approximates the integral \int_a^b f(x) dx by the sum:
    (dx/3) \sum_{k=1}^{N/2} (f(x_{2i-2} + 4f(x_{2i-1}) + f(x_{2i}))
    where x_i = a + i*dx and dx = (b - a)/N.
    Parameters
    ----------
    f : function
        Vectorized function of a single variable
    a , b : numbers
        Interval of integration [a,b]
    N : (even) integer
        Number of subintervals of [a,b]
    Returns
    -------
    float
        Approximation of the integral of f(x) from a to b using
        Simpson's rule with N subintervals of equal length.
    Examples
    --------
    >>> simps(lambda x : 3*x**2,0,1,10)
    1.0
    Notes:
    ------
    Stolen from: https://www.math.ubc.ca/~pwalls/math-python/integration/simpsons-rule/
    """
    if N % 2 == 1:
        raise ValueError("N must be an even integer.")
    dx = (b - a) / N
    x = jnp.linspace(a, b, N + 1)
    y = f(x)
    S = dx / 3 * jnp.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2], axis=0)
    return S