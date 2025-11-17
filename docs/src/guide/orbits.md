# Learning the Schwarzschild Metric

We will now discuss how our neural network seeks to learn the Schwarzschild Metric. The setup is as follows: The NN begins with the Newtonian weak-field limit metric as an ansatz and attempts to learn the Schwarzschild Metric by fitting the gravitational waveform. To demonstrate the scale of the problem, the (inverse) Newtonian Metric is as follows. We use the inverse metric (as opposed to the standard metric) by virtue of the way that we construct the Hamiltonian: 

$$g^{\mu\nu}_N = \begin{pmatrix} -\left(1-\frac{2}{r} \right)^{-1} & 0 & 0 & 0 \\ 
                        0 & \left(1+\frac{2}{r}\right)^{-1} & 0 & 0 \\ 
                        0 & 0 & 0 & 0 \\
                        0 & 0 & 0 & r^{-2} * \left(1 + \frac{2}{r} \right)^{-1}
\end{pmatrix}$$

The goal is to learn the (inverse) Schwarzschild metric: 

$$g^{\mu\nu}_S = \begin{pmatrix} -\left(1-\frac{2}{r} \right)^{-1} & 0 & 0 & 0 \\ 
                        0 & \left(1-\frac{2}{r}\right)^{+1} & 0 & 0 \\ 
                        0 & 0 & 0 & 0 \\
                        0 & 0 & 0 & r^{-2}
\end{pmatrix}$$

Now that we've set the stage, we see that the temporal component of both metrics $g^{tt}$ are the same. The neural network is tasked only with learning $g^{rr}$ and $g^{\phi\phi}$. How hard exactly is this problem? We can plot the radial and angular components for both metrics as a function of radius to see exactly what the NN has to learn: 

![RadialComponent](RadialComponent_Metric.png‎)

![AngularComponent](AngularComponent_Metric.png‎)

We will now construct our metric as follows, with the intention of the neural network learning 4 quantities:

> - Initial Conditions: The Neural Network will learn additive corrections to the initial conditions $(E,L)$ of the binary black holes
> - Metric Components: The Neural Network will learn multiplicative corrections to the metric components $g^{tt}$ and $g^{rr}$ of the Newtonian weak-field limit metric (and thereby attempt to learn the Schwarzschild metric)

We will now provide a full, step-by-step breakdown of our code for how we will achieve the above aims, with the ultimate result being that we will generate the metric fits for the following two orbits: $(p=15, e=0.5)$ and $(p=15, e=0.9)$. Here are the steps we will take: 

> - Step 0: Generate Training Data
> - Step 1: Import Training Data
> - Step 2: Define Simulation Parameters
> - Step 3: Define Initial Conditions
> - Step 4: Create Neural Networks
> - Step 5: Initialize Neural Networks
> - Step 6: Assign NN Inputs & Extract Outputs
> - Step 6A: Create Helper Function to Construct Initial Conditions
> - Step 7: Create Function for ODE Model
> - Step 8: Define & Solve ODE Model & Convert Orbit to Waveform
> - Step 8.1: Test Initial Solution
> - Step 9: Define Loss Function
> - Step 10: Run BFGS Optimization Algorithm

I will take you through this entire pipeline step-by-step. First, we begin with Step 0: Generating the training data. We will do this for both $(p=15, e=0.5)$ and $(p=15, e=0.9)$.

!!! note "Step 0: Generate Training Data"
    In our `BareBonesSchwarzschild.jl` code, we set the following parameters: `p = 15, e = 0.5`. To rapidly iterate and test our code, I will use a very small `datasize=100`, which calls for a correspondingly small timespan `tspan = (0, 2e3)`. Otherwise, if we choose a longer timespan, the orbit will be very choppy, because 100 data points will be spread thin and not sample the data as densely. 
    ![BlackHolesp15e05](BlackHoles_Orbit_p15_e05.png‎)
    ![GWp1505](GW_p15_e05.png‎)

!!! note "Step 1: Import Training Data"
    In our `FullNewtonian2Schwarzschild.jl` code, we import our true orbits and waveforms as simply as:
    ```
      x_ecc, y_ecc = file2trajectory(tsteps,"input/trajectoryA_Schwarzschild_p15_e0p5.txt")
      x2_ecc, y2_ecc = file2trajectory(tsteps,"input/trajectoryB_Schwarzschild_p15_e0p5.txt")
      waveform_real_ecc = file2waveform(tsteps,"input/waveform_real_Schwarzschild_p15_e0p5.txt")
      waveform_imag_ecc = file2waveform(tsteps,"input/waveform_imag_Schwarzschild_p15_e0p5.txt")
    ```

!!! note "Step 2: Define Simulation Parameters"
    We define our simulation parameters simply as follows. The masses are given by 
    ```
        mass_ratio = 1
        model_params = [mass_ratio]
        mass1 = 1.0/(1.0+mass_ratio)
        mass2 = mass_ratio/(1.0+mass_ratio)
    ```
    We can also define the timesteps and timespan as follows
    ```
      tspan = (0, 1.9e3)
      datasize = 100
      tsteps = range(tspan[1], tspan[2], length = datasize) 
      dt_data = tsteps[2] - tsteps[1]
      dt = 1.0
      num_optimization_increments = 20
    ```

!!! note "Step 3: Define Initial Conditions"
    We define our initial conditions as Newtonian initial conditions, as follows: 
    ```
        p = 15
        e = 0.5
        r_min = p / (1+e)
        r_max = p / (1-e)
        const rvals_penalty = range(r_min, r_max; length = 100)
        E0_base, L0_base = eccentric_pt_L(p, e) # Newtonian IC
     ```

!!! note "Step 4: Create Neural Networks"
    Our neural network is relatively simple, with just a single hidden layer: 
    ```
        NN_Conservative = Chain(
          Dense(1, 10, tanh),
          Dense(10, 10, tanh),
          Dense(10, 2),
        )
    ```

!!! note "Step 5: Initialize Neuarl Networks"
    We define the NN parameters and state as follows: `NN_Conservative_params, NN_Conservative_state = Lux.setup(rng, NN_Conservative)`. Now, we initialize the weights and biases of the NN near zero (near Newtonian weak-field limit conditions):
    ```
        for (i, layer) in enumerate(NN_Conservative_params)
            if ~isempty(layer)
                if i == length(NN_Conservative_params)  # Final layer
                    layer.weight .= 0
                    layer.bias .= 0 # Force output near 0
                else  # Hidden layers
                    layer.weight .= 0.1 * randn(rng, eltype(layer.weight), size(layer.weight))
                    layer.bias .= 0.1 * randn(rng, eltype(layer.bias), size(layer.bias))
                end
            end
        end
    ```

!!! note "Step 6: Assign NN Inputs & Extract Outputs"
    The only input to our neural network is the radial variable $r$ (In our state vector, that's `u[2]` because `u = [t, r, \theta, \phi, p_t, p_r, p_\theta, p_\phi]`. Thus, we create a function that's able to feed $r$ as the only input to the NN and extract the NN output: 
    ```
    function NN_adapter(u, params)
      conservative_features = [u[2]]
      conservative_output, _ = NN_Conservative(conservative_features, params.conservative, NN_Conservative_state)    
      return (conservative = conservative_output)
    end
    ```
    This adapter function will be used later to define our ODE model via `du = GENERIC(du, u, model_params, t, NN=NN_adapter_dual, NN_params=p)`. We also define a function `NN_params` to store all the corrections our NN will be making (i.e., store all the NN parameters): 
    ```
    NN_params = ComponentArray(
      conservative = NN_Conservative_params,
      dE0 = 0.0, 
      dL0 = 0.0
    )
    ```

!!! note "Step 6A: Create Helper Function to Construct Initial Conditions"
    Now we create a helper function which will add the NN corrections to the newtonian initial conditions and then construct the new state vector `u0` after these corrections: 
    ```
      function make_u0(params)
        E0 = E0_base + params.dE0
        L0 = L0_base + params.dL0
        return [
            0.0, # t
            r0, # r
            pi/2, # θ
            0.0, # ϕ
            E0, # pₜ
            0.0, # pᵣ
            0.0, # p_θ
            -L0, # p_ϕ
            0.0
        ]
      end
    ```

!!! note "Step 7: Create Function for ODE Model"
    We define our ODE model quite simply, as follows: 
    ```
    function ODE_model(du, u, p, t)
      du = GENERIC(du, u, model_params, t,
                                      NN=NN_adapter_dual, 
                                      NN_params=p)
      return du
    end
    ```

!!! note "Step 8: Define & Solve ODE Model & Convert Orbit to Waveform"
    We now solve the initial ODE problem, obtain the equations of motion, integrate them to obtain the orbits $(r(t), \phi(t))$, and then convert these orbits into the gravitational wave $h(t)$. 
    ```
    u0_init = make_u0(NN_params)
    prob_nn = ODEProblem(ODE_model, u0_init, tspan, NN_params)
    soln_nn = Array(solve(prob_nn_dual, Tsit5(), saveat = tsteps, dt = dt, adaptive=false, verbose=false))
    waveform_nn_real, waveform_nn_imag = compute_waveform(dt_data, soln_nn, mass_ratio; coorbital=false)
    orbit = soln2orbit(soln_nn)
    pred_orbit1_init, pred_orbit2_init = one2two(orbit, 1, mass_ratio)
    ```
    And now we plot our initial solution:
    ```
    plot(x_ecc, y_ecc, aspect_ratio=:equal, linewidth = 2, label = "Real")
    plot!(pred_orbit1_init[1,:], pred_orbit1_init[2,:], aspect_ratio=:equal, linewidth = 2, label = "Prediction")
    plot(waveform_real_ecc, label = "Real")
    plot!(waveform_nn_real, label = "Prediction")
    plot(waveform_imag_ecc, label = "Real")
    plot!(waveform_nn_imag, label = "Prediction")
    ```

If we now plot our results, they look as follows. 

![TrainingResultsp15](TrainingResults_p15_e0.5.png‎)










