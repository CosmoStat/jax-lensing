from jax_lensing.samplers.score_samplers import ScoreHamiltonianMonteCarlo, ScoreMetropolisAdjustedLangevinAlgorithm
from jax_lensing.samplers.tempered_sampling import TemperedMC
import numpy as np
import jax.numpy as jnp
import jax
import tensorflow_probability as tfp; tfp = tfp.substrates.jax

def tempered_HMC(init_image,
                 total_score_fn,
                 batch_size,
                 initial_step_size,
                 initial_temperature,
                 min_steps_per_temp,
                 num_results,
                 num_burnin_steps
                 ):
  def make_kernel_fn(target_log_prob_fn, target_score_fn, sigma):
    return ScoreHamiltonianMonteCarlo(
      target_log_prob_fn=target_log_prob_fn,
      target_score_fn=target_score_fn,
      step_size=initial_step_size*(jnp.max(sigma)/initial_temperature)**0.5,
      #step_size=10*(np.max(sigma)/s0)**0.5,
      num_leapfrog_steps=3,
      num_delta_logp_steps=4)

  tmc = TemperedMC(
            target_score_fn=total_score_fn,#score_prior,
            inverse_temperatures=initial_temperature*np.ones([batch_size]),
            make_kernel_fn=make_kernel_fn,
            gamma=0.98,
            min_steps_per_temp=min_steps_per_temp,
            num_delta_logp_steps=4)


  num_results = int(6e3)
  num_burnin_steps = int(0)


  samples, trace = tfp.mcmc.sample_chain(
        num_results=3,
        current_state=init_image.reshape([batch_size, -1]),
        kernel=tmc,
        num_burnin_steps=num_burnin_steps,
        num_steps_between_results=num_results//3,
        trace_fn=lambda _, pkr: (pkr.pre_tempering_results.is_accepted,
                                 pkr.post_tempering_inverse_temperatures,
                                 pkr.tempering_log_accept_ratio),
        seed=jax.random.PRNGKey(0))
 
  return samples, trace
