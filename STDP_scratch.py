import numpy as np

# From the paper 
a_plus = 0.004
a_minus = 0.003

t = 1 # current time

## Example spike time tensor S
S0 = np.array([[0, 1], [1, 0]])
S1 = np.array([[1, 0], [0, 0]])
S2 = np.array([[0, 0], [0, 1]])
S = np.array([[S0, S1, S2]]) # spike time tensor

## Convert spike time tensor into ST, an array which stores the spike time indices from S
ST = np.ndarray.flatten(np.zeros(np.shape(S[0,0,:,:])))
for x in range(np.shape(S)[1]):
    S_flat = np.ndarray.flatten(S[0,x,:,:])
    for n in range(len(S_flat)):
        if S_flat[n] > 0:
            ST[n]=x

## Example weight matrix of presynaptic neurons
W = np.ones((2,2))
W = 0.5 * W 

# save current W shape
W_shape = np.shape(W)

W = np.ndarray.flatten(W)

for j in range(len(ST)): # for each presynaptic neuron
    if ST[j]-t <= 0:    # pre spike before post spike, causality
        W[j] = W[j] + a_plus * W[j]*(1-W[j])    # weight update with a_plus
    if ST[j]-t > 0:     # post spike before pre spike, non-causality
        W[j] = W[j] + a_minus * W[j]*(1-W[j])   # weight update with a_minus

W = W.reshape(W_shape) # return W to its original shape













