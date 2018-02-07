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
    for i, x in enumerate(X):
        h = x @ W
        good_xe = np.e ** h[y[i]]
        rest_xe = np.sum(np.e ** h)
        loss_tmp = -np.log(good_xe / rest_xe)
        loss += loss_tmp

        grad_tmp = x / rest_xe
        for j in range(W.shape[1]):
            dW[:, j] += grad_tmp * (np.e ** h[j])
        dW[:, y[i]] -= x

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    loss /= X.shape[0]
    loss += reg * np.sum(W ** 2)
    dW /= X.shape[0]
    dW += reg * 2 * W
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
    H = X @ W
    XE = np.e ** H
    true_indices = np.arange(y.shape[0]), y
    good_XE = XE[true_indices]
    XE_sums = XE.sum(axis=1)
    loss = (-np.log(good_XE / XE_sums)).sum() / X.shape[0] + reg * np.sum(W ** 2)

    grad_tmp = (X.T / XE_sums)
    # print(grad_tmp.shape, XE.shape)
    dW += (grad_tmp @ XE)
    for i in range(W.shape[1]):
        # print(X[np.argwhere(y == i)].sum(axis=0).squeeze().shape, dW.shape)
        dW[:, i] -= X[np.argwhere(y == i)].sum(axis=0).squeeze()

    dW /= X.shape[0]
    dW += reg * 2 * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
