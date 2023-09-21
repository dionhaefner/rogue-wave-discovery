from typing import Tuple
from functools import partial
import math

import numpy as np
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_flatten

import flax.linen as nn
from flax.training import train_state

import optax


class MultiHeadMLP(nn.Module):
    features: Tuple[int]
    hidden_layers: Tuple[int]
    base_rate: float

    def bias_init(self, *args):
        return jnp.array([logit(self.base_rate)])

    def setup(self):
        self.bias = self.param("bias", self.bias_init, (1,), "float32")

    def __call__(self, x_in):
        components = self.eval_heads(x_in)
        return sum(components) + self.bias

    @nn.compact
    def eval_heads(self, x_in):
        out = []
        for head_feat in self.features:
            x = x_in[:, head_feat]
            for layer in self.hidden_layers:
                x = nn.relu(nn.Dense(layer)(x))

            x = nn.Dense(1, use_bias=False)(x)
            out.append(x)

        return out


@partial(jax.jit, static_argnames=("batch_size", "apply_fn"))
def batched_apply(state, features, batch_size, apply_fn=None):
    """Trade some speed for memory efficiency"""
    num_batches = math.ceil(features.shape[0] / batch_size)

    if apply_fn is None:
        apply_fn = state.apply_fn

    out = jnp.empty((features.shape[0], 1))

    def loop_body(i, arr):
        in_i = jax.lax.dynamic_slice_in_dim(
            features, i * batch_size, batch_size, axis=0
        )
        out_i = apply_fn({"params": state.params}, in_i)
        return jax.lax.dynamic_update_slice_in_dim(arr, out_i, i * batch_size, axis=0)

    return jax.lax.fori_loop(0, num_batches, loop_body, out)


def logit(prob):
    return jax.scipy.special.logit(prob)


def inverse_logit(logit):
    return jax.nn.sigmoid(logit)


def find_params_by_node_name(params, node_name):
    from typing import Mapping

    def _is_leaf_fun(x):
        if isinstance(x, Mapping) and jax.tree_util.all_leaves(x.values()):
            return True
        return False

    def _get_key_finder(key):
        def _finder(x):
            if not isinstance(x, Mapping):
                return None
            value = x.get(key)
            return None if value is None else {key: value}

        return _finder

    filtered_params = jax.tree_map(
        _get_key_finder(node_name), params, is_leaf=_is_leaf_fun
    )
    filtered_params = [
        x for x in jax.tree_util.tree_leaves(filtered_params) if x is not None
    ]

    return filtered_params


@jax.jit
def cross_entropy_regularized(logits, labels, params, irm_weight=0, l2_reg=0, l1_reg=0):
    def cross_entropy(logits, labels):
        logits = jnp.squeeze(logits)
        loss = labels * jax.nn.log_sigmoid(logits) + (1 - labels) * jax.nn.log_sigmoid(
            -logits
        )
        return -jnp.mean(loss)

    def invariance(logits, labels):
        def lossgrad(logits, labels):
            return jax.grad(lambda w: cross_entropy(w * logits, labels))(1.0)

        return lossgrad(logits[::2], labels[::2]) * lossgrad(logits[1::2], labels[1::2])

    classification_loss = cross_entropy(logits, labels)
    invariance_loss = irm_weight * invariance(logits, labels)

    l1_loss = 0.0
    l2_loss = 0.0

    kernel_params = find_params_by_node_name(params, "kernel")
    for param in kernel_params:
        l1_loss += jnp.sum(jnp.abs(param))
        l2_loss += jnp.sum(param**2)

    l1_loss = l1_reg * l1_loss
    l2_loss = l2_reg * jnp.sqrt(l2_loss)

    loss = classification_loss + invariance_loss + l1_loss + l2_loss
    return loss


@jax.jit
def score(state, x, y, logits=None):
    base_rate = jnp.sum(y) / len(y)
    if logits is None:
        logits = batched_apply(state, x, 1024)[:, 0]
    ignorance = y * jax.nn.log_sigmoid(logits) + (1 - y) * jax.nn.log_sigmoid(-logits)
    ignorance = jnp.mean(ignorance)
    ignorance_base = base_rate * jnp.log(base_rate) + (1 - base_rate) * jnp.log(
        1 - base_rate
    )
    return ignorance - ignorance_base


def train_swag(
    swag_state,
    x_train,
    y_train,
    loss_fn,
    num_steps,
    accumulate_every,
    max_cols_deviation,
    batch_size=1024,
    quiet=False,
):
    """Running variance, Knuth style.

    https://www.johndcook.com/blog/standard_deviation/
    """
    swag_param_mean = swag_param_mean_old = swag_state.params
    swag_param_welford_s = tree_map(jnp.zeros_like, swag_state.params)

    deviation = []

    progress = tqdm(range(1, num_steps + 1), disable=quiet)
    n_models = 1

    for epoch in progress:
        swag_state, train_loss = train_epoch(
            swag_state, x_train, y_train, batch_size, loss_fn=loss_fn, quiet=quiet
        )
        if not quiet:
            progress.write(
                f"epoch: {epoch: 3d}/{num_steps}, train_loss: {train_loss:.5f}"
            )

        if epoch % accumulate_every == 0:
            swag_param_mean = tree_map(
                lambda x, mean_old: mean_old + (x - mean_old) / n_models,
                swag_state.params,
                swag_param_mean_old,
            )

            swag_param_welford_s = tree_map(
                lambda x, s, mean_old, mean_new: s + (x - mean_old) * (x - mean_new),
                swag_state.params,
                swag_param_welford_s,
                swag_param_mean_old,
                swag_param_mean,
            )

            swag_param_mean_old = swag_param_mean

            n_models += 1

            deviation.append(
                tree_map(lambda x, y: x - y, swag_state.params, swag_param_mean)
            )
            if len(deviation) > max_cols_deviation:
                deviation = deviation[1:]

    cov_diag = tree_map(
        lambda s: jnp.maximum(0, s / (n_models - 1)), swag_param_welford_s
    )
    deviation = tree_map(lambda *args: jnp.stack(args, axis=-1), *deviation)
    return swag_state, (swag_param_mean, cov_diag, deviation)


def sample_swag(state, x, swag_mean, cov_diag, deviation, num_samples):
    k = tree_flatten(deviation)[0][0].shape[-1]

    out = np.empty((x.shape[0], num_samples))
    for i in range(num_samples):
        z2 = jnp.array(np.random.normal(size=(k,)))

        def sample_leaf(swag_mean, cov_diag, deviation):
            z1 = np.random.normal(size=swag_mean.shape)
            return (
                swag_mean
                + 1 / np.sqrt(2) * jnp.sqrt(cov_diag) * z1
                + 1 / np.sqrt(2 * (k - 1)) * jnp.dot(deviation, z2)
            )

        param_sample = tree_map(sample_leaf, swag_mean, cov_diag, deviation)
        sample_state = state.replace(params=param_sample)
        pred_sample = batched_apply(sample_state, x, 1024)
        out[:, i] = np.array(pred_sample[:, 0])

    return out


@partial(jax.jit, static_argnames=("loss_fn",))
def get_model_grad(state, features, labels, loss_fn):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss(params):
        logits = state.apply_fn({"params": params}, features)
        return loss_fn(logits, labels, params)

    grad_fn = jax.value_and_grad(loss)
    loss, grads = grad_fn(state.params)
    return grads, loss


def create_train_state(model, learning_rate, num_features):
    """Creates initial `TrainState`."""
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.ones([1, num_features]))["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@partial(jax.jit, static_argnames=("loss_fn",))
def train_multi_step(state, x_train, y_train, perms, loss_val, loss_fn):
    def loop_body(i, inputs):
        state, loss_val = inputs
        perm = perms[i]
        batch_data = x_train[perm, ...]
        batch_labels = y_train[perm, ...]
        grads, new_loss = get_model_grad(state, batch_data, batch_labels, loss_fn)
        state = state.apply_gradients(grads=grads)
        loss_val += new_loss
        return (state, loss_val)

    return jax.lax.fori_loop(0, perms.shape[0], loop_body, (state, loss_val))


def train_epoch(state, x_train, y_train, batch_size, loss_fn, quiet=False):
    """Train for a single epoch."""
    train_ds_size = len(x_train)
    steps_per_epoch = train_ds_size // batch_size

    perms = jnp.array(np.random.permutation(len(x_train)))
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    num_multisteps = 100
    num_iterations = math.ceil(steps_per_epoch / num_multisteps)
    loss = jnp.zeros(1)

    progress = tqdm(range(num_iterations), leave=False, disable=quiet)

    for i in progress:
        perm_slice = slice(
            i * num_multisteps, min((i + 1) * num_multisteps, steps_per_epoch)
        )
        state, loss = train_multi_step(
            state, x_train, y_train, perms[perm_slice], loss, loss_fn
        )

    loss /= steps_per_epoch
    return state, float(loss)


def train(
    model,
    x_train,
    y_train,
    loss_fn,
    x_val=None,
    y_val=None,
    batch_size=1024,
    num_epochs=100,
    learning_rate=1e-2,
    initial_state=None,
    quiet=False,
):
    num_features = x_train.shape[1]

    state = create_train_state(model, learning_rate, num_features)

    if initial_state is not None:
        state = state.replace(params=initial_state.params)

    progress = tqdm(range(1, num_epochs + 1), disable=quiet)

    try:
        for epoch in progress:
            state, train_loss = train_epoch(
                state, x_train, y_train, batch_size, loss_fn=loss_fn, quiet=quiet
            )

            train_score = score(state, x_train, y_train)

            if x_val is not None:
                val_score = score(state, x_val, y_val)
            else:
                val_score = 0

            if not quiet:
                progress.write(
                    f"epoch: {epoch: 3d}/{num_epochs}, train_loss: {train_loss:.5f}, "
                    f"train_score: {train_score:.5f}, val_score: {val_score:.5f}"
                )
    except KeyboardInterrupt:
        pass

    return state


def get_subset_models(model, sub_xy, loss_fn, *, learning_rate, epochs, swag_epochs):
    subset_states = {}
    subset_swag_samples = {}

    for subset, (sub_x, sub_y) in tqdm(sub_xy.items()):
        sub_state = train(
            model,
            sub_x,
            sub_y,
            num_epochs=epochs,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            quiet=True,
        )
        sub_state, swag_out_sub = train_swag(
            sub_state,
            sub_x,
            sub_y,
            loss_fn=loss_fn,
            num_steps=swag_epochs,
            accumulate_every=1,
            max_cols_deviation=30,
            quiet=True,
        )
        subset_states[subset] = sub_state
        subset_swag_samples[subset] = sample_swag(sub_state, sub_x, *swag_out_sub, 100)

    return subset_states, subset_swag_samples
