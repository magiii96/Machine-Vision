# Machine Vision Neural Network tutorial---Part 1: crossentropy_softmax_layer
# Author: Daniel E. Worrall, 3 Dec 2016
#
# This script contains the class definition for the crossentropy and
# softmax layer combined. This is a common techinque, used to avoid 
# numerical overflow in backpropagation. It contains two functions 
# 'forward' and 'backward'. These compute the forward- and back-propagation 
# steps respectively. You will need to fill out the sections marked 'TODO'.
import numpy as np

class Crossentropy_softmax_layer(object):
    # The properties section lists the variables associated with this layer
    # which are stored whenever the forward or backward methods are called.
    
    def __init__(self):
        self.softmax_output = 0  # Softmax output of network
        self.target = 0          # Targets
        self.x = 0               # Input
        self.y = 0               # Output loss

    def forward(self, x, target):
        # Compute the forward-propagated activations of this layer. You
        # can do this by computing the softmax followed by the
        # crossentropy.

        # subtract amax(x, axis=1) for stability
        x_stable = x - np.amax(x, axis=1, keepdims=True)
        self.softmax_output = softmax(x_stable)

        # TODO: Compute the crossentropy
        y = -np.sum(target * np.log(self.softmax_output)) / x.shape[0]
        
        # Store the input, output and target to selfect
        self.x = x
        self.y = y
        self.target = target
        return y

    def backward(self, dLdy):
        # Compute the back-propagated gradients of this layer.
        # Note that the softmax contains no parameters, so dLdW is
        # empty. Also note that the input gradient dLdy is a scalar.

        # TODO: Compute the gradients wrt the input. (Cheatsheet)
        dLdx = self.softmax_output - self.target
        
        return dLdx

    
def softmax(x):
    # TODO: Compute softmax output
    y = np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)
    return y

