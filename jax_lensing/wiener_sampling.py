import sys
sys.path = ['../'] + sys.path
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax_lensing.spectral import make_power_map
from jax_lensing.inversion import ks93inv, ks93
import numpy as np
import gc
import jax
from jax import lax

def spin_wiener_filter(data_q, data_u, ncov_diag_Q,ncov_diag_U, input_ps_map_E, input_ps_map_B, iterations=10):
    """
    Wiener filter Elsner-Wandelt messenger field adapted for spin-2 fields (CMB polarization or galaxy weak lensing(
    Parameters
    ----------
    data_q : Q square image data (e.g. gamma1)
    data_u : U square image data (e.g. gamma2)
    ncov_diag_Q : Q noise variance per pixel (assumed uncorrelated)
    ncov_diag_U : U noise variance per pixel (assumed uncorrelated)
    input_ps_map_E : 1D power P(k) for E-mode signal power spectrum evaluated 2D components k1,k2 as a square image
    input_ps_map_B : 1D power P(k) for B-mode signal power spectrum evaluated 2D components k1,k2 as a square image
    iterations : number of iterations

    Returns
    -------
    s_q,s_u : Wiener filtered q and u signals
    """
    tcov_diag = jnp.min(jnp.array([ncov_diag_Q, ncov_diag_U]))
    scov_ft_E = jnp.fft.fftshift(input_ps_map_E)
    scov_ft_B = jnp.fft.fftshift(input_ps_map_B)
    s_q = jnp.zeros(data_q.shape)
    s_u = jnp.zeros(data_q.shape)

    for i in jnp.arange(iterations):
        # in Q, U representation
        t_Q  = (tcov_diag/ncov_diag_Q)*data_q + ((ncov_diag_Q-tcov_diag)/ncov_diag_Q) * s_q
        t_U  = (tcov_diag/ncov_diag_U)*data_u + ((ncov_diag_U-tcov_diag)/ncov_diag_U) * s_u
        # in E, B representation
        t_E, t_B = ks93(t_Q,t_U)
        s_E  = (scov_ft_E/(scov_ft_E+tcov_diag))*jnp.fft.fft2(t_E)
        s_B  = (scov_ft_B/(scov_ft_B+tcov_diag))*jnp.fft.fft2(t_B)
        s_E = jnp.fft.ifft2(s_E)
        s_B = jnp.fft.ifft2(s_B)
        # in Q, U representation
        s_q, s_u = ks93inv(s_E,s_B)

    return s_q, s_u



def spin_wiener_sampler(data_q, data_u, ncov_diag_Q,ncov_diag_U, input_ps_map_E, input_ps_map_B, iterations=10, initial_map=None, thinning=1, verbose=False):
    """
    Wiener posterior sampler using Elsner-Wandelt messenger field adapted for spin-2 fields (CMB polarization or galaxy weak lensing(
    Parameters
    Parameters
    ----------
    data_q : Q square image data (e.g. gamma1)
    data_u : U square image data (e.g. gamma2)
    ncov_diag_Q : Q noise variance per pixel (assumed uncorrelated)
    ncov_diag_U : U noise variance per pixel (assumed uncorrelated)
    input_ps_map_E : 1D power P(k) for E-mode signal power spectrum evaluated 2D components k1,k2 as a square image
    input_ps_map_B : 1D power P(k) for B-mode signal power spectrum evaluated 2D components k1,k2 as a square image
    iterations : number of iterations
    initial_map : starting image for the sampler
    thinning : thinning factor (iterations must be divisible by thinning factor)
    verbose : bool verbose

    Returns
    -------
    samples_E, samples_B : samples from Wiener posterior
    """
    size = (data_q).shape[0]
    tcov_diag = np.min(np.array([ncov_diag_Q, ncov_diag_U]))
    tcov_ft = tcov_diag # unnecessary really, but convention dependent
    scov_ft_E = np.fft.fftshift(input_ps_map_E)
    scov_ft_B = np.fft.fftshift(input_ps_map_B)
    sigma_t_squared_Q =  tcov_diag  - tcov_diag * tcov_diag / ncov_diag_Q
    sigma_t_squared_U =  tcov_diag  - tcov_diag * tcov_diag / ncov_diag_U
    sigma_s_squared_E =  scov_ft_E*tcov_ft/(tcov_ft+scov_ft_E)
    sigma_s_squared_B =  scov_ft_B*tcov_ft/(tcov_ft+scov_ft_B)
    
    print(sigma_s_squared_B.mean())

    if initial_map is None:
        s = data_q + 1j*data_u
    else:
        s = np.copy(initial_map)

    assert (iterations%thinning == 0)
    
    samples_E = np.zeros(shape=(int(iterations/thinning), size,size), dtype=jnp.complex128)
    samples_B = np.zeros(shape=(int(iterations/thinning), size,size), dtype=jnp.complex128)
    
    for i in range(iterations):
        # in Q, U representation
        t_Q = (tcov_diag/ncov_diag_Q)*data_q + ((ncov_diag_Q-tcov_diag)/ncov_diag_Q) * s[0]
        t_U = (tcov_diag/ncov_diag_U)*data_u + ((ncov_diag_U-tcov_diag)/ncov_diag_U) * s[1]
        t_Q = np.random.normal(t_Q.real, np.sqrt(sigma_t_squared_Q.real))
        t_U = np.random.normal(t_U.real, np.sqrt(sigma_t_squared_U.real))
        # in E, B representation
        t = ks93(t_Q,t_U)
        s_E = (scov_ft_E/(scov_ft_E+tcov_ft))*np.fft.fft2(t[0])
        s_B = (scov_ft_B/(scov_ft_B+tcov_ft))*np.fft.fft2(t[1])
        
        
        s_E = np.random.normal(s_E.real*0., np.sqrt(sigma_s_squared_E.real)*size) + s_E
        s_B = np.random.normal(s_B.real*0., np.sqrt(sigma_s_squared_B.real)*size) + s_B
        s_E = (np.fft.ifft2(s_E))
        s_E = (s_E.real + s_E.imag)
        s_B = (np.fft.ifft2(s_B))
        s_B = (s_B.real + s_B.imag)
        s = ks93inv(s_E,s_B)
        if i%thinning==0:
            samples_E[int(i/thinning)] = s_E
            samples_B[int(i/thinning)] = s_B
            if verbose==True:
                print(i)
    return samples_E, samples_B


# doesn't work :/
@jit
def spin_wiener_filter_jit(data_q, data_u, ncov_diag_Q,ncov_diag_U, input_ps_map_E, input_ps_map_B, iterations=10, verbose=False):
    """
    Similar to spin_wiener_filter, but with XLA acceleration
    """
    size = (data_q.real).shape[0]
    tcov_diag = jnp.min(jnp.array([ncov_diag_Q, ncov_diag_U]))
    tcov_ft = tcov_diag#/(size**2.)
    scov_ft_E = jnp.fft.fftshift(input_ps_map_E)
    scov_ft_B = jnp.fft.fftshift(input_ps_map_B)
    s_q = jnp.zeros(data_q.shape)
    s_u = jnp.zeros(data_q.shape)

    q_operationA = (tcov_diag/ncov_diag_Q)*data_q
    u_operationA = (tcov_diag/ncov_diag_U)*data_u

    q_operationB = ((ncov_diag_Q-tcov_diag)/ncov_diag_Q)
    u_operationB = ((ncov_diag_U-tcov_diag)/ncov_diag_U)

    def body_fun(i, val):
        s_q, s_u = val

        t_Q  = q_operationA + jnp.multiply(q_operationB,s_q)
        t_U  = u_operationA + jnp.multiply(u_operationB,s_u)
        # in E, B representation
        t_E, t_B = ks93(t_Q,t_U)
        s_E  = (scov_ft_E/(scov_ft_E+tcov_ft))*jnp.fft.fft2(t_E)
        s_B  = (scov_ft_B/(scov_ft_B+tcov_ft))*jnp.fft.fft2(t_B)
        s_E = jnp.fft.ifft2(s_E)
        s_B = jnp.fft.ifft2(s_B)
        # in Q, U representation
        s_q, s_u = ks93inv(s_E,s_B)
   
        return (s_q, s_u)
    
    (s_q, s_u) = lax.fori_loop(0, iterations, body_fun, (s_q, s_u))

    return s_q,s_u
