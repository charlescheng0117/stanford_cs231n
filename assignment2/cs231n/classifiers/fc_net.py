from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        W1 = np.random.normal(loc=0.0, scale=weight_scale, size=(input_dim, hidden_dim)) 
        b1 = np.zeros((hidden_dim))

        W2 = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, num_classes))
        b2 = np.zeros((num_classes))
        
        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        z1, cache_f1 = affine_forward(X, self.params['W1'], self.params['b1'])
        a1, cache_relu1 = relu_forward(z1)

        z2, cache_f2 = affine_forward(a1, self.params['W2'], self.params['b2'])
        scores = z2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss_data, dscores = softmax_loss(scores, y)
        
        # backward
        da1, dW2, db2 = affine_backward(dscores, cache_f2) 

        dz1 = relu_backward(da1, cache_relu1)

        dX, dW1, db1 = affine_backward(dz1, cache_f1)

        grads['W1'], grads['b1'], grads['W2'], grads['b2'] = dW1, db1, dW2, db2
        # gradients from reg loss
        grads['W1'] += self.reg * self.params['W1']
        grads['W2'] += self.reg * self.params['W2']

        # loss
        loss_reg = 0.5 * self.reg * (np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2)) 
        loss += loss_data + loss_reg

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        L = self.num_layers 
        in_dim = input_dim

        use_batchnorm = self.normalization == "batchnorm"
        # if L = 4 => we have hidden_dims = [H1, H2, H3]
        # then the nn is
        # x ======> H1 ======> H2 ======> H3 ======> y
        #   W1, b1     W2, b2     W3, b3     W4, b4
        for i in range(L-1):
            out_dim = hidden_dims[i]
            self.params['W{}'.format(i+1)] = np.random.normal(loc=0.0, scale=weight_scale, size=(in_dim, out_dim))
            self.params['b{}'.format(i+1)] = np.zeros((out_dim))

            if self.normalization is not None:
                self.params['gamma{}'.format(i+1)] = np.ones((out_dim))
                self.params['beta{}'.format(i+1)] = np.zeros((out_dim))

            in_dim = out_dim
        
        # output layer
        in_dim = hidden_dims[-1]
        self.params['W{}'.format(L)] = np.random.normal(loc=0.0, scale=weight_scale, size=(in_dim, num_classes))
        self.params['b{}'.format(L)] = np.zeros((num_classes))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        
        # {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
        L = self.num_layers
        a_prev = X
        cache = {}
        use_batchnorm = self.normalization == 'batchnorm'
        use_layernorm = self.normalization == 'layernorm'
        
        for i in range(1, L):
            Wi, bi = self.params['W{}'.format(i)], self.params['b{}'.format(i)]
            # affine forward
            z_fci, cache_fci = affine_forward(a_prev, Wi, bi)

            if self.normalization is not None:
                # batchnorm forward
                if use_batchnorm:
                    gamma_i, beta_i = self.params['gamma{}'.format(i)], self.params['beta{}'.format(i)]
                    zi, cache_bni = batchnorm_forward(z_fci, gamma_i, beta_i, self.bn_params[i-1])
                    cache['bn{}'.format(i)] = cache_bni
                # layernorm forward
                else:
                    gamma_i, beta_i = self.params['gamma{}'.format(i)], self.params['beta{}'.format(i)]
                    zi, cache_lni = layernorm_forward(z_fci, gamma_i, beta_i, self.bn_params[i-1])
                    cache['ln{}'.format(i)] = cache_lni
            else:
                zi = z_fci
            
            # relu forward
            a_ri, cache_ri = relu_forward(zi)
            
            if self.use_dropout:
                ai, cache_doi = dropout_forward(a_ri, self.dropout_param)
                cache['do{}'.format(i)] = cache_doi
            else:
                ai = a_ri
            
            cache['fc{}'.format(i)] = cache_fci # cache for affine_forward
            cache['r{}'.format(i)] = cache_ri # cache for relu_forward

            a_prev = ai

        # output layer
        WL, bL = self.params['W{}'.format(L)], self.params['b{}'.format(L)]
        scores, cache_fL = affine_forward(a_prev, WL, bL)
        cache['fc{}'.format(L)] = cache_fL

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0.0

        da_prev, dWL, dbL = affine_backward(dscores, cache['fc{}'.format(L)])
        grads['W{}'.format(L)] = dWL + self.reg * self.params['W{}'.format(L)]
        grads['b{}'.format(L)] = dbL
        reg_loss += 0.5 * self.reg * np.sum( self.params['W{}'.format(L)] ** 2 )

        for i in reversed(range(1, L)):
            # relu backward
            drelu_i = relu_backward(da_prev, cache['r{}'.format(i)])

            # dropout backward
            if self.use_dropout:
                cache_doi = cache['do{}'.format(i)] 
                ddropout_i = dropout_backward(drelu_i, cache_doi)
                
                drelu_i =ddropout_i
            
            if self.normalization is not None:
                # batchnorm backward
                if use_batchnorm:
                    dzi, dgamma, dbeta = batchnorm_backward(drelu_i, cache['bn{}'.format(i)])
                    grads['gamma{}'.format(i)] = dgamma
                    grads['beta{}'.format(i)] = dbeta
                # layernorm backward
                else:
                    dzi, dgamma, dbeta = layernorm_backward(drelu_i, cache['ln{}'.format(i)])
                    grads['gamma{}'.format(i)] = dgamma
                    grads['beta{}'.format(i)] = dbeta
            else:
                dzi = drelu_i

            da_prev, dWi, dbi = affine_backward(dzi, cache['fc{}'.format(i)])
            grads['W{}'.format(i)] = dWi + self.reg * self.params['W{}'.format(i)]
            grads['b{}'.format(i)] = dbi
            reg_loss += 0.5 * self.reg * np.sum( self.params['W{}'.format(i)] ** 2 )

        loss += data_loss + reg_loss

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that perorms an affine transform followed by 
    bn - ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma, beta, bn_param: params for batchnorm_forward

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    a_bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_param) 
    out, relu_cache = relu_forward(a_bn)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache

def affine_bn_relu_backward(dout, cache):
    """
    Backward pass for the affine-bn-relu convenience layer
    """
    fc_cache, bn_cache, relu_cache = cache
    da_bn = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = batchnorm_backward(da_bn, bn_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta
