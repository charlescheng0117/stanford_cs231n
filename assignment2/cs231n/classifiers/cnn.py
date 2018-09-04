from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #                           
        ############################################################################
        C, H, W = input_dim
        W1 = np.random.normal(loc=0.0, scale=weight_scale, size=(num_filters, C, filter_size, filter_size))
        b1 = np.zeros((num_filters,))
        
        W2 = np.random.normal(loc=0.0, scale=weight_scale, size=(num_filters * (H // 2) * (W // 2), hidden_dim)) # 2x2 max pool, so H' => H//2, W' => W//2
        b2 = np.zeros((hidden_dim,))

        W3 = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, num_classes))
        b3 = np.zeros((num_classes))
        
        self.params['W1'], self.params['b1'] = W1, b1
        self.params['W2'], self.params['b2'] = W2, b2
        self.params['W3'], self.params['b3'] = W3, b3

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
       
        use_fast = True
        if use_fast:
            conv_forward_naive = conv_forward_fast
            conv_backward_naive = conv_backward_fast
            max_pool_forward_naive = max_pool_forward_fast
            max_pool_backward_naive = max_pool_backward_fast

        # conv - relu - 2x2 max pool - affine - relu - affine - softmax
        conv1, cache_conv1 = conv_forward_naive(X, W1, b1, conv_param)
        relu1, cache_relu1 = relu_forward(conv1)
        pool1, cache_pool1 = max_pool_forward_naive(relu1, pool_param)

        fc2, cache_fc2 = affine_forward(pool1, W2, b2)
        relu2, cache_relu2 = relu_forward(fc2)

        fc3, cache_fc3 = affine_forward(relu2, W3, b3)
        scores = fc3 

        data_loss, dscores = softmax_loss(scores, y)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        reg = self.reg
        reg_loss = 0.0
        # backward
        drelu2, dW3, db3 = affine_backward(dscores, cache_fc3)
        
        dfc2 = relu_backward(drelu2, cache_relu2)
        dpool1, dW2, db2 = affine_backward(dfc2, cache_fc2)

        drelu1 = max_pool_backward_naive(dpool1, cache_pool1)
        dconv1 = relu_backward(drelu1, cache_relu1)
        dX, dW1, db1 = conv_backward_naive(dconv1, cache_conv1)

        dW3 += reg * W3 
        dW2 += reg * W2
        dW1 += reg * W1

        reg_loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))
        
        loss = data_loss + reg_loss
        grads['W1'], grads['b1'] = dW1, db1
        grads['W2'], grads['b2'] = dW2, db2
        grads['W3'], grads['b3'] = dW3, db3
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
