import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]

  for i in range(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    correct_class_score = scores[y[i]]
    loss = loss - np.log( np.exp(correct_class_score) / np.sum(np.exp(scores)) )
    
    for j in range(num_class):
      dW[:,j] += np.exp(scores[j]) / np.sum(np.exp(scores)) * X[i] 
    dW[:, y[i]] -= X[i]

  loss = loss / num_train + reg * np.sum(W * W)
  dW = dW / num_train + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  mat_scores = X.dot(W)
  row_max_scores = np.max(mat_scores, axis=1) # shape: (N, )
  row_max_scores = row_max_scores.reshape( (num_train, -1) ) # reshape to (N, 1) for broadcasting
  mat_scores -= row_max_scores

  correct_class_scores = mat_scores[np.arange(num_train), y] # shape: (N, )
  correct_class_scores = correct_class_scores.reshape( (num_train, -1) )
  #loss = - np.sum ( np.log( np.exp(correct_class_scores) / np.sum( np.exp(mat_scores), axis=1) ) ) ??? will encounter => np.log(0)
  loss = - np.sum(correct_class_scores)  + np.sum(  np.log (np.sum( np.exp(mat_scores), axis=1) ) )

  #for j in range(num_class):
  #  dW[:,j] += np.exp(scores[j]) / np.sum(np.exp(scores)) * X[i] 
  cross_entropy = np.exp(mat_scores) / ( np.sum(np.exp(mat_scores), axis=1).reshape((num_train, 1)) )  
  # dW[:, y[i]] -= X[i]
  cross_entropy[np.arange(num_train), y] -= 1 

  dW += X.T.dot(cross_entropy)

  #for j in range(num_class):
  #  dW[:,j] += np.exp(scores[j]) / np.sum(np.exp(scores)) * X[i] 
  #dW[:, y[i]] -= X[i]

  loss = loss / num_train + reg * np.sum(W * W)
  dW = dW / num_train + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

