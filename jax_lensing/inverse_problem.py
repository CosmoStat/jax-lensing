import jax.numpy as np
from jax import random, grad, jit
from jax.experimental import optimizers
from jax_lensing.inversion import ks93, ks93inv

# Define the objective functions to minimise
@jit
def square_norm(kappa, g1, g2):
    kE, kB = kappa
    gamma1, gamma2 = ks93inv(kE, kB)
    return np.sum((g1-gamma1)*(g1-gamma1)) + np.sum((g2-gamma2)*(g2-gamma2))

@jit
def square_norm_smooth(kappa, g1, g2, reg=1.):
    kE, kB = kappa
    gamma1, gamma2 = ks93inv(kE, kB)
    sq = np.sum((g1-gamma1)*(g1-gamma1)) + np.sum((g2-gamma2)*(g2-gamma2))
    p = np.sum(kE*kE) + np.sum(kB*kB)
    return sq + reg * p

@jit
def square_norm_sparse(kappa, g1, g2, reg=.01):
    kE, kB = kappa
    gamma1, gamma2 = ks93inv(kE, kB)
    sq = np.sum((g1-gamma1)*(g1-gamma1)) + np.sum((g2-gamma2)*(g2-gamma2))
    p = np.sum(np.abs(kE)) + np.sum(np.abs(kB))
    return sq + reg * p

def gamma2kappa(g1, g2, kappa_shape, obj, step_size=0.01, n_iter=1000):

    # Set up optimizer
    init_kE = np.zeros(kappa_shape)
    init_kB = np.zeros(kappa_shape)
    init_params = (init_kE, init_kB)
    opt_init, opt_update, get_params = optimizers.sgd(step_size=0.001)
    opt_state = opt_init(init_params)

    @jit
    def update(i, opt_state):
        params = get_params(opt_state)
        gradient = grad(obj)(params, g1, g2)
        return opt_update(i, gradient, opt_state)

    # Loop
    for t in range(n_iter):
        opt_state = update(t, opt_state)
        params = get_params(opt_state)

    kEhat, kBhat = params
    return kEhat, kBhat

if __name__=='__main__':

    key = random.PRNGKey(0)

    # (g1, g2) should in practice be measurements from a real galaxy survey
    g1, g2 = 0.1 * random.normal(key, (2, 32, 32)) + 0.1 * np.ones((2, 32, 32))
    kE, kB = ks93(g1, g2)

    def least_squares(g1, g2, kE, kB):
        gamma1, gamma2 = ks93inv(kE, kB)
        return np.linalg.norm(np.vstack([g1, g2]) - np.vstack([gamma1, gamma2]))

    # Computing shear from convergence
    gamma1, gamma2 = ks93inv(kE, kB)
    print('gamma1.shape', gamma1.shape)
    print('gamma error', least_squares(g1, g2, kE, kB))

    # Computing convergence form shear
    kappaE, kappaB = ks93(gamma1, gamma2)
    print('kappaE.mean()', kappaE.mean(), 'should be 0')
    print('kappa error', np.linalg.norm(kE-kappaE))

    print('')
    print('Recovering kappa with SGD')

    kEhat, kBhat = inverse_problem(g1, g2, obj=square_norm_smooth, kappa_shape=kE.shape)
    print(kEhat.shape)
    print(np.mean(kEhat))
