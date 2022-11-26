import hard_spheres_utils as hs_utils
import jax.numpy as jnp
from jax import random
import numpy as np

n_dimensions=2
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
    if i%20 == 0:
        positions_all.append(positions)


np.savetxt("trajectory.txt",positions_all)



