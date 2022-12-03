import hard_spheres_utils as hs_utils
import jax.numpy as jnp
from jax import random
import numpy as np
import argparse
import math

'''n_dimensions=2
n_particles=16
box_vectors = jnp.asarray([[1.1*n_particles,0],[0,1.1*n_particles]])
radius=1.0

positions_init=jnp.asarray([[x*1.05,y*1.05] for x in range(4) for y in range(4)],dtype=float)


n_iterations=5000
key=random.PRNGKey(42)
positions_all=[positions_init]
positions=positions_init
for i in range(n_iterations):
    positions,key=hs_utils.single_step(positions,n_particles,box_vectors,key)
    if i%(2*n_particles) == 0:
        positions_all.append(positions)


np.savetxt("trajectory.txt",positions_all)'''



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_dimensions", type=int, default=2)
    parser.add_argument("--n_particles", type=int, default=16)
    parser.add_argument("--box_vectors", required=False, type=str)
    parser.add_argument("--n_steps", type=int, default=10000)

    args = parser.parse_args()
    n_dimensions = args.n_dimensions
    n_particles = args.n_particles
    if args.box_vectors:
        box_vectors = jnp.asarray(args.box_vectors)
    else:
        box_vectors = jnp.asarray([[1.1*n_particles,0],[0,1.1*n_particles]])
    n_steps = args.n_steps

    positions_init=jnp.asarray([[x*1.05,y*1.05] for x in range(int(math.sqrt(n_particles))) for y in range(int(math.sqrt(n_particles)))],dtype=float)
    key=random.PRNGKey(42)
    positions_all=[positions_init]
    positions=positions_init
    for i in range(n_steps):
        positions,key=hs_utils.single_step(positions,n_particles,box_vectors,key)
        if i%(2*n_particles) == 0:
            positions_all.append(positions)

    np.save("positions.npy",positions_all)