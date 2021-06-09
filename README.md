# Sample from hard sphere configurations using score based generative models


## What does this code do?

This code uses jax to sample configurations of hard spheres, and builds a model for the score (the gradient of the data distribution), which can then be used to sample from the distribution using langevin dynamics

