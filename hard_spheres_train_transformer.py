import jax
from jax import vmap, jit, grad
import jax.numpy as jnp
from jax import random
import numpy as np
from hard_spheres_utils import *
import numpy as onp
import matplotlib.pyplot as plt
from simple_transformer import TransformerEncoder

from jax.example_libraries import optimizers
from jax.example_libraries import stax
from functools import partial
import argparse
import json

'''n_particles = 4
n_dimensions = 2
batch_size = 3
box_vectors = jnp.asarray([[1.1*n_particles,0],[0,1.1*n_particles]])
positions, box_vectors = hard_spheres_init(n_dimensions = n_dimensions, n_particles = n_particles, box_vectors = box_vectors, radius=1.0)

#intialize a transfromer encoder
main_rng = random.PRNGKey(42)
input_encoding_dimension = 2 #same as the number of dimensions of the particle positions

encoder = TransformerEncoder(num_layers=4,
                              input_dim=input_encoding_dimension, #particle positions for the simple case, same as n_dimensions
                              num_heads=1,
                              dim_feedforward=56,
                              dropout_prob=0.0)

main_rng, x_rng = random.split(main_rng)
x = random.normal(x_rng, (n_particles, input_encoding_dimension))
main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
params = encoder.init({'params': init_rng, 'dropout': dropout_init_rng}, x, train=True)['params']



# Create optimizer. Note that both network and optimizer returns pure (stateless) functions
opt_init, opt_update, get_params = optimizers.adam(1e-3)
'''

@jax.jit
def compute_loss(net_params, inputs):
    # v-- a function that computes jacobian by forward mode differentiation
    jacobian = jax.jacfwd(encoder.apply, argnums=-1)
    
    # we use jax.vmap to vectorize jacobian function along batch dimension
    batch_jacobian = jax.vmap(partial(jacobian, {'params': params}))(inputs)  # [batch, dim, dim]
    #I don't know if the dimensions are correct here
    trace_jacobian = jnp.trace(batch_jacobian, axis1=1, axis2=2)
    output_norm_sq = jnp.square(jax.vmap(partial(encoder.apply,{'params': params}))(inputs))
    
    return jnp.mean(trace_jacobian + 1/2 * output_norm_sq)


@jax.jit
def train_step(step_i, opt_state, batch):
    net_params = get_params(opt_state)
    loss = compute_loss(net_params, batch)
    grads = jax.grad(compute_loss, argnums=0)(net_params, batch)
    return loss, opt_update(step_i, grads, opt_state)


'''opt_state = opt_init(params)


loss_history=[]
batch=sample_batch(positions, n_particles, box_vectors, batch_size=batch_size)
for i in range(2000):
    print("training %d"%i)
    batch = sample_batch(batch[-1].reshape([4,2]), n_particles, box_vectors, batch_size=batch_size)
    loss, opt_state = train_step(i, opt_state, batch)
    loss_history.append(loss.item())
    print(loss,i)'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory', type=str, required=True)
    parser.add_argument('--encoder_layers', type=int, default=4, help='Number of layers in the encoder', required=False)
    parser.add_argument('--encoder_heads', type=int, default=1, help='Number of heads in the encoder', required=False)
    parser.add_argument('--encoder_dim_feedforward', type=int, default=56, help='Number of hidden units in the encoder', required=False)

    args = parser.parse_args()
    
    input_directory = args.input_directory
    #read input json file to a dictionary from the output directory:
    with open(input_directory + '/args.json') as f:
        hard_sphere_params = json.load(f)
    

    input_encoding_dimension = hard_sphere_params['n_dimensions'] #same as the number of dimensions of the particle positions
    n_particles = hard_sphere_params['n_particles']

    encoder = TransformerEncoder(num_layers = args.encoder_layers,
                                input_dim = input_encoding_dimension, #particle positions for the simple case, same as n_dimensions
                                num_heads=args.encoder_heads,
                                dim_feedforward=args.encoder_dim_feedforward,
                                dropout_prob=0.1)

    main_rng = random.PRNGKey(42)
    main_rng, x_rng = random.split(main_rng)
    x = random.normal(x_rng, (n_particles, input_encoding_dimension))

    main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
    params = encoder.init({'params': init_rng, 'dropout': dropout_init_rng}, x, train=True)['params']
    opt_init, opt_update, get_params = optimizers.adam(1e-2)
    opt_state = opt_init(params)
    n_batches = int(hard_sphere_params['n_steps']/hard_sphere_params['batch_size'])
    loss_history=[]
    for i in range(n_batches):
        print("training %d"%i)
        batch = np.load(input_directory + '/positions_%d.npy'%i)
        loss, opt_state = train_step(i, opt_state, batch)
        loss_history.append(loss.item())
        print(loss,i)