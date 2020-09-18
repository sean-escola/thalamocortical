import numpy as np
import numpy.random as npr

from time import time

import tensorflow as tf
import tensorflow_probability as tfp

# a stable version of 1/x
def _reciprocal(x, ep=1e-20):
    return x / (x * x + ep)

# wrapper of numpy's eig() with custom gradient
@tf.custom_gradient
def _myeig(A):
    e, v = np.linalg.eig(A)
    def grad(grad_e, grad_v):
        f = _reciprocal(e[..., None, :] - e[..., None])
        f = tf.linalg.set_diag(f, tf.zeros_like(e))
        f = tf.math.conj(f)
        vt = tf.linalg.adjoint(v)
        vgv = vt @ grad_v
        mid = tf.linalg.diag(grad_e) + f * (vgv - vt @ (v * tf.linalg.diag_part(vgv)[..., None, :]))
        grad_a = tf.linalg.solve(vt, mid @ vt)
        return tf.cast(grad_a, A.dtype)
    return (e, v), grad

# wrapper of Tensorflow's py_function(), which let's us use our custom op
def myeig(A):
    if A.dtype == tf.float32:
        return tf.py_function(func=_myeig, inp=[A], Tout=[tf.complex64, tf.complex64])
    elif A.dtype == tf.float64:
        return tf.py_function(func=_myeig, inp=[A], Tout=[tf.complex128, tf.complex128])
    else:
        raise NotImplementedError('Expecting tf.float32 or tf.float64')

def build_model(N, n_fracs, K, tau, alpha, betas):
    ns = np.round(N * np.atleast_1d(n_fracs)).astype(int)
    betas = np.atleast_1d(betas)
    batch_shape=(K, ns.size, betas.size)
    J0s = npr.randn(*batch_shape, N, N) / np.sqrt(N)
    ws = npr.randn(*batch_shape, N) / np.sqrt(N)
    sigma_U = sigma_V = 1 / np.sqrt(N)  # default scaling
    Us = npr.randn(*batch_shape, N, ns.max()) * sigma_U
    Vs = npr.randn(*batch_shape, ns.max(), N) * sigma_V
    masks = np.ones_like(Us)
    for i, n in enumerate(ns):
        masks[:, i, ..., n:] = 0
    return dict(N=N,
                n=ns.max(),
                ns=ns, 
                J0s=J0s, 
                ws=ws, 
                sigma_U=sigma_U, 
                sigma_V=sigma_V, 
                Us=Us, 
                Vs=Vs, 
                masks=masks,
                tau=tau,
                alpha=alpha,
                betas=tf.cast(tf.broadcast_to(betas, batch_shape), tf.complex128),
                batch_shape=batch_shape
               )

# param is a 1-d tensor; reshape to N x n x 2 tensor, then extract U & V.T; renormalize U & V.T if needed; return U and V
def extract_Us_and_Vs(param, model):
    param = tf.reshape(param, (*param.shape[:-1], model['N'], model['n'], 2))
    Us, Vs = param[..., 0], param[..., 1]
    Us = (Us / tf.linalg.norm(Us, axis=-2, keepdims=True)) * model['masks'] * np.sqrt(model['N']) * model['sigma_U']
    Vs = (Vs / tf.linalg.norm(Vs, axis=-2, keepdims=True)) * model['masks'] * np.sqrt(model['N']) * model['sigma_V']
    return Us, tf.linalg.matrix_transpose(Vs)

# get J0 + UV from param and J0
def _extract_Js(param, model):
    Us, Vs = extract_Us_and_Vs(param, model)
    return model['J0s'] + Us @ Vs

# get loss from eigenvalues d and eigenvectors r and other parameters
def _loss_from_eigs(d, r, model):
    L = tf.linalg.inv(tf.linalg.adjoint(r) @ r)
    g = (d - 1) / model['tau']
    g_real = tf.math.real(g)
    g_column = g[..., None]
    g_row = tf.math.conj(g)[..., None, :]
    F = -_reciprocal(g_column + g_row)
    G = (g_column * g_row) * F
    wR = tf.linalg.matvec(r, tf.cast(model['ws'], tf.complex128), transpose_a=True)
    return tf.math.real(tf.reduce_sum((r @ (L * F)) * tf.math.conj(r), axis=[-1, -2]) / model['N'] / model['tau'] + model['betas'] * model['tau'] * tf.reduce_sum(wR * tf.linalg.matvec(L * G, tf.math.conj(wR)), axis=-1)) \
           + model['alpha'] * tf.reduce_sum(tf.where(g_real < 0, tf.zeros_like(g_real), g_real)**2, axis=-1)

# return loss (or batch of losses)
def loss_fx(param, model):
    Js = _extract_Js(param, model)
    d, r = myeig(Js)
    return _loss_from_eigs(d, r, model)

def _make_param(model, param=None):
    if param is None:
        param = np.stack((model['Us'], np.swapaxes(model['Vs'], -2, -1)), axis=-1)
        param = param.reshape(*model['batch_shape'], -1)
        param = tf.constant(param)
    return param

def do_adam(model, epochs, param=None, losses=None, learning_rate=0.01, print_interval=1, verbosity=1):
    losses = [] if losses is None else [*losses]
    param = tf.Variable(_make_param(model, param))

    if verbosity: print(f'Starting ADAM optimization')
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    cur = _cur = time()
    # initial loss value
    with tf.GradientTape() as tape:
        tape.watch(param)
        tmp = loss_fx(param, model)
        loss = tf.reduce_mean(tmp)
    losses.append(tmp.numpy())
    
    # run ADAM
    for i in range(epochs):
        if verbosity > 1 and i % print_interval == 0:
            print(f"   epoch: {i:3d}, time: {time() - _cur:6.2f} s, mean loss: {losses[-1].mean():8f}")
            _cur = time()
        grad = tape.gradient(loss, param)
        opt.apply_gradients(zip([grad], [param]))
        with tf.GradientTape() as tape:
            tape.watch(param)
            tmp = loss_fx(param, model)
            loss = tf.reduce_mean(tmp)
        losses.append(tmp.numpy())
    if verbosity > 1: print(f"   epoch: {epochs:3d}, time: {time() - _cur:6.2f} s, mean loss: {losses[-1].mean():8f}")

    if verbosity:
        print(f'ADAM completed')
        print(f'   total time: {time() - cur:.2f} s')
        print(f'   num epochs: {epochs:d}')
        print(f'   mean loss: {losses[-1].mean():.8f}')
        if verbosity > 2: print(f'   loss array:\n{losses[-1]}')

    return tf.constant(param.numpy()), losses

def do_lbfgs(model, iters, param=None, losses=None, print_interval=1, verbosity=1, **kwargs):
    losses = [] if losses is None else [*losses]
    param = _make_param(model, param)

    def _loss_for_lbfgs(param):
        _model = dict(model)
        if len(_model['batch_shape']) > 0:
            tmp = tf.math.is_finite(param)
            assert tf.reduce_all(tf.reduce_all(tmp, axis=-1) | (tf.reduce_any(tmp, axis=-1) == False)), "we expect nan's to be present or absent by batch"
            idx = tmp[..., 0]
            if not tf.reduce_all(idx):
                param = tf.boolean_mask(param, idx)
                _model['J0s'] = tf.boolean_mask(_model['J0s'], idx)
                _model['ws'] = tf.boolean_mask(_model['ws'], idx)
                _model['masks'] = tf.boolean_mask(_model['masks'], idx)
                _model['betas'] = tf.boolean_mask(_model['betas'], idx)
        l, g = tfp.math.value_and_gradient(lambda x: loss_fx(x, _model), param)
        if len(_model['batch_shape']) > 0 and not tf.reduce_all(idx):
            idx = tf.reshape(idx, -1)
            nans = tf.ones(idx.shape, dtype=tf.float64) * np.nan
            l = tf.where(idx, tf.scatter_nd(tf.where(idx), l, idx.shape), nans)
            l = tf.reshape(l, _model['batch_shape'])
            g = tf.where(idx[:, None], tf.scatter_nd(tf.where(idx), g, [idx.shape[0], g.shape[-1]]), nans[:, None])
            g = tf.reshape(g, [*_model['batch_shape'], -1])
        return l, g

    if verbosity: print(f'Starting L-BFGS optimization')
    cur  = _cur = time()
    # initialize results object
    results = tfp.optimizer.lbfgs_minimize(
            _loss_for_lbfgs,
            initial_position=param,
            max_iterations=0,
            **kwargs
        )
    losses.append(results.objective_value.numpy())

    # run one step of L-BFGS at a time
    for i in range(iters):
        if verbosity > 1 and i % print_interval == 0:
            print(f"   iter: {i:3d}, fn_evals: {results.num_objective_evaluations:4d}, time: {time() - _cur:6.2f} s, mean loss: {losses[-1].mean():8f}")
            _cur = time()
        results = tfp.optimizer.lbfgs_minimize(
            _loss_for_lbfgs,
            initial_position=None,
            previous_optimizer_results=results,
            max_iterations=results.num_iterations + 1,
            **kwargs
        )
        losses.append(results.objective_value.numpy())
        if np.all(results.converged | results.failed):
            break
    if verbosity > 1: print(f"   iter: {results.num_iterations:3d}, fn_evals: {results.num_objective_evaluations:4d}, time: {time() - _cur:6.2f} s, mean loss: {losses[-1].mean():8f}")

    if verbosity:
        print(f'L-BFGS completed')
        print(f'   total time: {time() - cur:.2f} s')
        print(f'   num iters: {results.num_iterations:d}')
        print(f'   num func evals: {results.num_objective_evaluations:d}')
        print(f'   frac converged: {results.converged.numpy().mean():.3f}')
        print(f'   frac failed: {results.failed.numpy().mean():.3f}')
        print(f'   mean loss: {results.objective_value.numpy().mean():.8f}')
        print(f'   loss array:\n{results.objective_value.numpy().flatten()}')

    return results.position, losses, results
