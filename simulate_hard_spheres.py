import hard_spheres_utils as hs_utils
import jax.numpy as jnp
from jax import random
import numpy as np
import argparse
import math
import json
from hexalattice.hexalattice import create_hex_grid
import os



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_dimensions", type=int, default=2)
    parser.add_argument("--n_particles", type=int, default=16)
    parser.add_argument("--box_vectors", required=False, type=str)
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--prng_key", required=False, type=int, default=42)
    parser.add_argument("--output_directory", required=False, type=str, default="output")
    parser.add_argument("--max_particle_size", required=False, type=float, default=0.95)

    args = parser.parse_args()
    n_dimensions = args.n_dimensions
    n_particles = args.n_particles
    if args.box_vectors:
        box_vectors = jnp.asarray(args.box_vectors)
    else:
        box_vectors = jnp.asarray([[1.0*math.sqrt(n_particles),0],[0,1.0*math.sqrt(n_particles)]])
    n_steps = args.n_steps
    batch_size = args.batch_size
    prng_key = args.prng_key
    output_directory = args.output_directory
    #check if output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    #write the args to a json file:
    with open("{}/args.json".format(output_directory), 'w') as fp:
        json.dump(args.__dict__, fp, indent=4)
    positions_init = jnp.asarray(create_hex_grid(int(math.sqrt(n_particles)),int(math.sqrt(n_particles)),align_to_origin=False)[0])
    sizes_init = jnp.asarray([args.max_particle_size for i in range(n_particles)])
    key=random.PRNGKey(42)
    positions=positions_init
    sizes=sizes_init
    for i in range(int(n_steps/batch_size)):
        batch_positions = hs_utils.sample_batch(positions_init,n_particles,box_vectors,batch_size,sizes)
        sizes=sizes*0.98
        positions_init = batch_positions[-1]
        #drop the third row (which is the size) of the positions array
        positions_init = positions_init[:,:-1]
        np.save("{}/positions_{}.npy".format(args.output_directory,i),batch_positions)

