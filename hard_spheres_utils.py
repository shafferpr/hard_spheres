import jax
from jax import vmap, jit, grad
import jax.numpy as jnp
from jax import random
import numpy as np

n_dimensions=2
radius=0.95

@jax.jit
def absminND(a, axis=None):
    amax = a.max(axis)
    amin = a.min(axis)
    return jnp.where(-amin < amax, amin, amax)

@jax.jit
def absmin(a):
    return a[jnp.abs(a).argmin()]

@jax.jit
def distance(x,y,box_v): #this computes a periodic distance between two points in the space defined by the box vectors box_v
    non_periodic_distance = x-y
    displacement_pairs=jnp.concatenate([box_v,-1*box_v, jnp.asarray([[0,0]])])
    all_images=non_periodic_distance+displacement_pairs
    x=jnp.asarray([absmin(all_images[:,i]) for i in range(n_dimensions)])
    return jnp.linalg.norm(x)

@jax.jit
def distance_pbc(x,y,box_v):
    #find the distance using periodic boundary conditions between the points x and y within the parallelogram defined by the vectors box_v[0] and box_v[1]
    inv_basis = jnp.linalg.inv(box_v)
    coord1 = jnp.dot(inv_basis, x)
    coord2 = jnp.dot(inv_basis, y)
    diff = coord2 - coord1
    diff -= jnp.round(diff)
    periodic_diff = jnp.dot(box_v, diff)
    return jnp.linalg.norm(periodic_diff)

@jax.jit
def nonoverlapping(x,y,box_v):
    dist=distance_pbc(x,y,box_v)
    return  jnp.where(dist>radius, True, False)

@jax.jit
def overlapping(x, y ,box_v):
    p1, r1 = x[:n_dimensions], x[-1] #split x into position and radius
    p2, r2 = y[:n_dimensions], y[-1] #split y into position and radius
    radius = (r1+r2)/2
    dist=distance_pbc(p1,p2,box_v)
    return  jnp.where(dist<radius, 1, 0)

batched_distance=vmap(distance,in_axes=(0,None,None))

batched_nonoverlaps=vmap(nonoverlapping,in_axes=(0,None,None))

batched_overlaps=jit(vmap(overlapping,in_axes=(0,None,None)))

def hard_spheres_init(n_dimensions=2, n_particles=4, box_vectors=jnp.asarray([[1.1*4,0],[0,1.1*4]]), radius=1.0):
    positions=jnp.asarray([[x*1.05,y*1.05] for x in range(int(np.sqrt(n_particles))) for y in range(int(np.sqrt(n_particles)))],dtype=float)
    return positions, box_vectors

@jax.jit
def displace_particle(positions, n_particles, box_vectors, key):
    key, subkey = jax.random.split(key)
    particle_index=random.randint(shape=(),minval=0,maxval=n_particles,key=subkey)
    displacement=0.05*random.normal(key=subkey, shape=[n_dimensions])
    new_position = apply_periodic_bc(box_vectors, positions[particle_index], displacement)
    return new_position, particle_index, key

@jax.jit
def apply_periodic_bc(box_vectors, x, dx):
    inv_basis = jnp.linalg.inv(box_vectors)
    coord = jnp.dot(inv_basis, x)
    coord += jnp.dot(inv_basis, dx)
    coord -= jnp.floor(coord)
    new_x = jnp.dot(box_vectors, coord)
    return new_x


@jax.jit
def single_step(positions, n_particles, box_vectors, key, sizes):
    new_position, particle_index, key=displace_particle(positions, n_particles, box_vectors, key)
    size = sizes[particle_index]
    overlaps=batched_overlaps(jnp.concatenate([positions,jnp.asarray(sizes.reshape(-1,1))], axis=1), jnp.concatenate([new_position,jnp.asarray([size])]), box_vectors)
    n_overlaps = jnp.count_nonzero(overlaps)
    new_positions = jnp.where(n_overlaps == 1, positions.at[particle_index].set(new_position), positions)
    return new_positions, key
    #if n_overlaps == 1:
    #    new_positions = positions.at[particle_index].set(new_position)
    #    return new_positions, key
    #else:
    #    return positions, key
    


def sample_batch(pos_init, n_particles, box_vectors, batch_size, sizes):
    positions=pos_init
    batch=[]
    for j in range(batch_size):
        n_iterations=400
        key=random.PRNGKey(j)
        for i in range(n_iterations):
            positions,key=single_step(positions, n_particles, box_vectors, key, sizes)
        #append sizes vector to positions
        positions_and_sizes = jnp.concatenate([positions, jnp.asarray(sizes).reshape(-1,1)], axis=1)
        batch.append(positions_and_sizes)
    return jnp.asarray(batch)
    
