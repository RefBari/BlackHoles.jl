# Fundamentals of Neural Networks

We will be using the `Lux.jl` library to create neural networks in Julia. Here's what we'll need to import: 

```julia
using Lux # to create our neural network
using Random # to initialize our network
using Zygote # automatic differentiation engine in Julia (used for backpropagation)
using Optimisers # provides gradient descent algorithm
using Plots # to create plots
using Distributions # to create a toy dataset
using Statistics # provides the mean function
```

Let's now define the hyperparameters of the network:

```julia
N_SAMPLES = 200 # number of data samples for our toy dataset
LAYERS = [1, 10, 10, 10, 1] # 1D input --> 1D output; 3 hidden layers with 10 dimensions
LEARNING_RATE = 0.1
N_EPOCHS = 30,000
```

Now we need a psuedo-random number generator

```julia
rng = Xoshiro(42)
```

Now we can produce our toy dataset:

```julia
x_samples = rand( # produce random samples
                 rng, # provide random # generator
                 Uniform(0, 2 * pi)) # define what distribution you want to sample from
                 (1, N_SAMPLES) # array that we want to sample; 1 is the spatial dimension and N_SAMPLES is the batch dimension
                )
```

Now let's create some noise for our data: 

```julia
y_noise = rand(
              rng, # random number generator
              Normal(0, 0.3) # Create noise from normal distribution with center at 0.0 and standard deviation of 0.3
              (1, N_SAMPLES) # Draw n by n sample values because we want to corrupt each sample value individually
              )
```
Now let's create a noisy sine function from these parts: 

```julia
y_samples = sin.(x_samples) .+ y_noise
```
Now we plot this by doing `scatter(x_samples[:], y_samples[:], label = "data")`. Now let's define the architechture of our neural network. Since we'll create a simple feedforward neural network, we'll express oits architechture in the form of a chain of functions: 

```julia
model = Chain( # NN = nested chain of functions
             [Dense(fan_in => fan_out, Lux.sigmoid) for (fan_in, fan_out) in zip(LAYERS[1:end-2], LAYERS[2:end-1])] ... # sigmoid activation function for the transition between each layer
             Dense(LAYERS[end-1] => LAYERS[end], identity) # last layer has no activation function, only first three layers has sigmoid activation
            )
```

In total, our network has 251 parameters. What does that mean? We will now initialize the parameters for the NN, as well as the layer states. 

```julia
parameters, layer_states = Lux.setup(rng, model)
```

Now let's make a prediction using the initial 250 parameters of the neural network.  
```julia
y_initial_prediction, layer_states = model(x_samples, parameters, layer_states) # carry the parameters and layer states of the NN through
```

Now let's see what the initial prediction of the NN is: 
```julia
scatter!(x_sample[:], y_initial_prediction[:], label = "initial prediction")
```
We will now define a loss function
```julia
function loss_fn(p, ls)
  y_prediction, new_ls = model(x_samples, p, ls)
  loss = 0.5 * mean((y_prediction .- y_samples).^2)
  return loss, new_ls
end
```

Create an optimizer which uses gradient descent (or you can use `ADAM` or `BFGS` if you'd like):
```julia
opt = Descent(LEARNING_RATE)
opt_state = Optimisers.setup(opt, parameters) # optimize / minimize the loss function over the parameters of the NN
```

Now, we define a training loop

```julia
loss_history = []
for epoch in 1:N_EPOCHS
  (loss, layer_states,), back = pullback(loss_fn, parameters, layer_states) # we use (loss, layer_states) to get the output of the loss function, which returns a tuple
  grad, _ = back((1.0, nothing))

  opt_state, parameters = Optimisers.update(opt_state, parameters, grad)
  push!(loss_history, loss)

  if epoch % 100 == 0
    println("Epoch: $Epoch, Loss: $Loss")
  end
end
```
Now we can plot the loss function by `plot(loss_history, yscale =:log10)`. 

We now see that the final prediction is much better than the initial one! 

```julia
y_final_prediction, layer_states = model(x_samples, parameters, layer_state)
scatter(x_samples[:], y_samples[:], label = "data")
scatter!(x_samples[:], y_final_prediction[:], label = "final prediction")
```









