# Toy Model

To validate whether the network is sufficiently expressive to represent the functions

$$g_{tt}, g_{rr}, g_{\theta\theta}, g_{\phi\phi}, g_{t\phi}$$

in the first place, we will create a simple proof-of-concept architechture with the following loss function: 

$$L=L_{g_{tt}} + L_{g_{rr}} + L_{g_{\theta\theta}} + L_{g_{\phi\phi}} + L_{g_{t\phi}}$$

Each of these loss terms will directly compare the true metric component against the predicted metric component: 

\begin{align}
L =    & |g^{\text{pred}}_{tt}-g^{\text{true}}_{tt}|^2 
  \\ + & |g^{\text{pred}}_{rr}-g^{\text{true}}_{rr}|^2
  \\ + & |g^{\text{pred}}_{\theta\theta}-g^{\text{true}}_{\theta\theta}|^2
  \\ + & |g^{\text{pred}}_{\phi\phi}-g^{\text{true}}_{\phi\phi}|^2
  \\ + & |g^{\text{pred}}_{t\phi}-g^{\text{true}}_{t\phi}|^2
\end{align}
