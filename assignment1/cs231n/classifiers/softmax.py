import numpy as np
from random import shuffle
from past.builtins import xrange

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
  N=y.shape[0]
  #D=W.shape[0]
  C=W.shape[1]
  
  global_loss=0
  for i in np.arange(N):
    local_numerator=0
    local_denominator=0
    for j in np.arange(C):#calculate denominator inside log
      local_exp=np.exp(np.dot(X[i,:], W[:,j]))
      local_denominator+=local_exp
    for j in np.arange(C):
      dW[:,j] += np.exp(np.dot(X[i,:], W[:,j])) * X[i,:] / local_denominator
    
    dW[:,y[i]] -= X[i,:] #gradient by numerator of log
    local_numerator=np.exp(np.matmul(X[i,:],W[:,y[i]]))#calculate numerator inside log
    
    global_loss+=-np.log(local_numerator/local_denominator)#calculate loss for one sample

  dW/=N
  #regularization term
  reg_term=np.sum(np.power(W,2)) * reg
  loss += global_loss/N + reg_term
  dW += 2 * reg * W
  pass
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
  N=y.shape[0]

  XW=np.matmul(X, W)
  exp_term=np.exp(XW)
  exp_sum=np.sum(exp_term, axis=1)

  loss = np.sum(-np.log(exp_term[np.arange(N),y]/ exp_sum))/N #loss term
  loss += np.sum(np.power(W,2)) * reg #reg loss term

  dW = np.matmul(X.T, exp_term/np.reshape(exp_sum, (-1,1)))/N
  binary=np.zeros_like(exp_term)
  binary[np.arange(N), y]=1
  dW -= np.matmul(X.T, binary)/N
  dW += 2 * reg * W #reg dW term

  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

