import jax
from jax import vmap, jit, grad
import jax.numpy as jnp
from jax import random
import numpy as np

'''n_dimensions=2
n_particles=100
box_vectors=jnp.asarray([[11,0],[0,11]])
radius=1.0'''

n_dimensions=2
n_particles=4
box_vectors=jnp.asarray([[2.2,0],[0,2.2]])
radius=1.0

positions=jnp.asarray([[x*1.05,y*1.05] for x in range(2) for y in range(2)],dtype=float)


@jax.jit
def absminND(a, axis=None):
    amax = a.max(axis)
    amin = a.min(axis)
    return jnp.where(-amin < amax, amin, amax)

@jax.jit
def absmin(a):
    return a[jnp.abs(a).argmin()]

@jax.jit
def distance(x,y,box_v):
    non_periodic_distance = x-y
    displacement_pairs=jnp.concatenate([box_v,-1*box_v, jnp.asarray([[0,0]])])
    all_images=non_periodic_distance+displacement_pairs
    #x=jnp.asarray([min(all_images[:,i],key=abs) for i in range(n_dimensions)])
    x=jnp.asarray([absmin(all_images[:,i]) for i in range(n_dimensions)])
    return jnp.sqrt(jnp.sum(jnp.square(x)))

@jax.jit
def nonoverlapping(x,y,box_v):
    dist=distance(x,y,box_v)
    return  jnp.where(dist>radius, True, False)

@jax.jit
def overlapping(x,y,box_v):
    dist=distance(x,y,box_v)
    return  jnp.where(dist<radius, True, False)

batched_distance=vmap(distance,in_axes=(0,None,None))

batched_nonoverlaps=vmap(nonoverlapping,in_axes=(0,None,None))

batched_overlaps=vmap(overlapping,in_axes=(0,None,None))

#@jax.jit
def single_step(positions, box_vectors, key):
    key, subkey = jax.random.split(key)
    particle_index=random.randint(shape=(),minval=0,maxval=100,key=subkey)
    displacement=0.05*random.normal(key=subkey,shape=[2])
    new_position=positions[particle_index]+displacement

    overlaps=batched_overlaps(positions,new_position,box_vectors)
    n_overlaps=len(list(filter(lambda x: x, overlaps)))
    if n_overlaps<2:
        new_positions=jax.ops.index_update(positions,particle_index,new_position)
        return new_positions, key
    else:
        return positions, key


n_iterations=5000
key=random.PRNGKey(42)
for i in range(n_iterations):
    positions,key=single_step(positions,box_vectors,key)

    
def sample_batch(pos_init, size):
    positions=pos_init
    batch=[]
    for j in range(size):
        n_iterations=200
        key=random.PRNGKey(j)
        for i in range(n_iterations):
            positions,key=single_step(positions,box_vectors,key)
        batch.append(positions.flatten())
    return jnp.asarray(batch)

import numpy as onp
import matplotlib.pyplot as plt

R_plt=onp.asarray(positions)

ms = 20


plt.plot(R_plt[:, 0], R_plt[:, 1], 'o', markersize=ms * 1.0)


plt.xlim([-0.5+onp.min(R_plt[:, 0]), onp.max(R_plt[:, 0])+0.5])
plt.ylim([-0.5+onp.min(R_plt[:, 1]), onp.max(R_plt[:, 1])+0.5])

plt.axis('off')

from jax.experimental import optimizers
from jax.experimental import stax
from functools import partial

# Set up network to predict scores

net_init, net_apply = stax.serial(
    stax.Dense(64), stax.Softplus,
    stax.Dense(164), stax.Softplus,
    stax.Dense(8),
)

# Create optimizer. Note that both network and optimizer returns pure (stateless) functions
opt_init, opt_update, get_params = optimizers.adam(1e-3)

# v-- jax.jit compiles a function for efficient CPU and GPU execution

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

def sample_batch(pos_init, size):
    positions=pos_init
    batch=[]
    for j in range(size):
        n_iterations=200
        key=random.PRNGKey(j)
        for i in range(n_iterations):
            positions,key=single_step(positions,box_vectors,key)
        batch.append(positions.flatten())
    return jnp.asarray(batch)

out_shape, net_params = net_init(jax.random.PRNGKey(seed=42), input_shape=(-1,8))
opt_state = opt_init(net_params)


loss_history=[]
batch=sample_batch(positions,size=64)
for i in range(2000):
    batch = sample_batch(batch[-1].reshape([4,2]),size=64)
    loss, opt_state = train_step(i, opt_state, x)
    loss_history.append(loss.item())
    print(loss,i)


def sample_langevin(x_initial, *, net_params, key, eps=1e-2, eps_decay=0.9, num_steps=15, temperature=1.0):
    """ sample x ~ p(x) by applying approximate Langvenin Dynamics, return a sequence of x_t """
    x_t, x_sequence = x_initial, [x_initial]
    
    for t in range(num_steps):
        key, subkey = jax.random.split(key)
        z_t = jax.random.normal(subkey, shape=x_t.shape)
        x_t = x_t + eps / 2 * net_apply(net_params, x_t) + jnp.sqrt(eps) * temperature * z_t
        x_sequence.append(x_t)
        eps *= eps_decay
        
    return jnp.stack(x_sequence)

key = jax.random.PRNGKey(42)
net_params = get_params(opt_state)

key, subkey = jax.random.split(key)
# sample x sequence
xx = sample_langevin(np.asarray([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]), key=subkey, net_params=net_params)
plt.scatter(xx.T[0], xx.T[1], color="blue")