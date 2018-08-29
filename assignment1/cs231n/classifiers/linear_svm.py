import numpy as np
from random import shuffle



def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        dW[:, j] += X[i]
        dW[:, y[i]] -= X[i] 
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train 

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  mat_scores = X.dot(W)
  correct_class_scores = mat_scores[np.arange(num_train), y] # indexing multi-dimensional arrays, shape: (N, )
  correct_class_scores_broad = np.reshape(correct_class_scores, (num_train, 1)) # reshape to (N, 1) for broadcasting
  mat_margins = mat_scores - correct_class_scores_broad + 1

  thresholded_margins = np.maximum(np.zeros(mat_margins.shape), mat_margins)
  thresholded_margins[np.arange(num_train), y] = 0 # correct_class don't contribute to loss
  loss = np.sum(thresholded_margins) / num_train + reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # this should be derived from paper and pencil
  mask_dW = (thresholded_margins > 0) * 1.0 # for dW[:, j] += X[i]
  
  # for dW[:, y[i]] -= X[i]
  # compute how many X[i] to deduct from dW[:, y[i]]
  #mask_dW_correct_class = np.sum(thresholded_margins, axis=1) * np.array(y) # shape: (N, )
  mask_dW_correct_class = np.sum(mask_dW, axis=1) # shape: (N, )
  mask_dW[np.arange(num_train), y] = -mask_dW_correct_class

  # shapes,
  # X.T:               (D, N)
  # mask_dW:           (N, C)
  # dot(X.T, mask_dW): (D, C)
  
  dW += X.T.dot(mask_dW)
  dW = dW / num_train + 2 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
