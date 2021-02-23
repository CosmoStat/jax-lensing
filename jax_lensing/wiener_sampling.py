from jax import grad, jit, vmap
# @jit
def spin_wiener_filter(data_q, data_u, ncov_diag_Q,ncov_diag_U, input_ps_map_E, input_ps_map_B, iterations=10, verbose=False):
    """
    From Elsner & Wandelt '13 (messenger field paper) with spin addition
    """
    size = (data_q.real).shape[0]
    tcov_diag = jnp.min(jnp.array([ncov_diag_Q, ncov_diag_U]))
    tcov_ft = tcov_diag#/(size**2.)
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
        s_E  = (scov_ft_E/(scov_ft_E+tcov_ft))*jnp.fft.fft2(t_E)
        s_B  = (scov_ft_B/(scov_ft_B+tcov_ft))*jnp.fft.fft2(t_B)
        s_E = jnp.fft.ifft2(s_E)
        s_B = jnp.fft.ifft2(s_B)
        # in Q, U representation
        s_q, s_u = ks93inv(s_E,s_B)

    return s_q,s_u
