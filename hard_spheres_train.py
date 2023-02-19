import jax
from jax import vmap, jit, grad
import jax.numpy as jnp
from jax import random
import numpy as np
from hard_spheres_utils import *
import numpy as onp
import matplotlib.pyplot as plt
from transformer import TransformerEncoder

from jax.example_libraries import optimizers
from jax.example_libraries import stax
from functools import partial
n_particles = 4
n_dimensions = 2
box_vectors = jnp.asarray([[1.1*n_particles,0],[0,1.1*n_particles]])
positions, box_vectors = hard_spheres_init(n_dimensions = n_dimensions, n_particles = n_particles, box_vectors = box_vectors,radius=1.0)


encoder = TransformerEncoder(num_layers=4,
                              input_dim=2, #particle positions
                              num_heads=3,
                              dim_feedforward=128,
                              dropout_prob=0.1)


# Set up network to predict scores

net_init, net_apply = stax.serial(
    stax.Dense(64), stax.Softplus,
    stax.Dense(164), stax.Softplus,
    stax.Dense(n_dimensions*n_particles),
)

# Create optimizer. Note that both network and optimizer returns pure (stateless) functions
opt_init, opt_update, get_params = optimizers.adam(1e-3)


@jax.jit
def compute_loss(net_params, inputs):
    # v-- a function that computes jacobian by forward mode differentiation
    jacobian = jax.jacfwd(net_apply, argnums=-1)
    
    # we use jax.vmap to vectorize jacobian function along batch dimension
    batch_jacobian = jax.vmap(partial(jacobian, net_params))(inputs)  # [batch, dim, dim]
    
    trace_jacobian = jnp.trace(batch_jacobian, axis1=1, axis2=2)
    output_norm_sq = jnp.square(net_apply(net_params, inputs)).sum(axis=1)
    
    return jnp.mean(trace_jacobian + 1/2 * output_norm_sq)


@jax.jit
def train_step(step_i, opt_state, batch):
    net_params = get_params(opt_state)
    loss = compute_loss(net_params, batch)
    grads = jax.grad(compute_loss, argnums=0)(net_params, batch)
    return loss, opt_update(step_i, grads, opt_state)


out_shape, net_params = net_init(jax.random.PRNGKey(seed=42), input_shape=(-1,n_dimensions*n_particles))
opt_state = opt_init(net_params)


loss_history=[]
batch=sample_batch(positions, n_particles, box_vectors, batch_size=64)
'''for i in range(2000):
    print("training %d"%i)
    batch = sample_batch(batch[-1].reshape([4,2]), n_particles, box_vectors, batch_size=64)
    loss, opt_state = train_step(i, opt_state, batch)
    loss_history.append(loss.item())
    print(loss,i)'''

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_directory', type=str, help='input directory')
    