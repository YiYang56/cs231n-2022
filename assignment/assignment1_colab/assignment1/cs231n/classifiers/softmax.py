from builtins import range
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, D, C = X.shape[0], X.shape[1], W.shape[1]
    for i in range(N):
      scores = X[i].dot(W)
      exp_score = np.exp(scores - np.max(scores)) # 减法是为了防止指数爆炸 - np.max(scores)
      loss += -np.log(exp_score[y[i]] / np.sum(exp_score))

      dexp_score = np.zeros_like(exp_score)
      dexp_score[y[i]] -= 1 / exp_score[y[i]]
      dexp_score += 1 / np.sum(exp_score)
      dscore = dexp_score * exp_score
      dW += X[[i]].T.dot([dscore])

    loss /= N
    loss += reg * np.sum(W * W)
    dW /= N
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    N, D, C = X.shape[0], X.shape[1], W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X @ W # N*C
    scores -= np.max(scores, axis=1, keepdims=True)
    loss1 = -scores[range(N), y] + np.log(np.sum(np.exp(scores), axis=1))
    loss = np.sum(loss1) / N + reg * np.sum(W * W)

    dloss1 = np.ones_like(loss1)
    dscores_loacl = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    dscores_loacl[range(N), y] -= 1
    dscores = dloss1.reshape(-1, 1) * dscores_loacl
    dW = X.T @ dscores / N + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
