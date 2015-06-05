import numpy as np
from random import shuffle
import pdb

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  num_classes = W.shape[0]
  num_train = X.shape[1]
  for i in xrange(num_train):
    # forward
    scores = W.dot(X[:,i])
    scores -= np.max(scores)
    numerator = np.exp(scores[y[i]]) 
    denominator = np.sum(np.exp(scores))
    prob = numerator / denominator
    loss += -1. * np.log(prob)

    # backward
    dscores = np.zeros(scores.shape)
    dloss = (1.)
    dprob = (-1.) / prob * dloss
    dnumerator = (1/denominator) * dprob
    ddenominator = (-1.*numerator) / (denominator**2) * dprob
    for j in xrange(num_classes):
      dscores[j] += np.exp(scores[j]) * ddenominator
    dscores[y[i]] += np.exp(scores[y[i]]) * dnumerator
    dW += dscores.reshape(len(dscores),1).dot(X[:,i].reshape(len(X[:,i]),1).T)
  
  loss /= num_train
  dW /= num_train
  
  loss += 0.5 * reg * np.sum(W*W)
  dW += (reg) * W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  num_classes = W.shape[0]
  num_train = X.shape[1]
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  # forward
  Scores = W.dot(X)
  Scores -= np.max(Scores)
  Numerator = np.exp(Scores[y,np.arange(num_train)])
  Denominator = np.sum(np.exp(Scores),axis=0)
  Prob = Numerator / Denominator
  loss += np.sum(-1.*np.log(Prob))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)
  
  # backward
  dScores = np.zeros(Scores.shape)
  dloss = 1.
  dProb = (-1.) / Prob * dloss
  dNumerator = (1./Denominator) * dProb
  dDenominator = (-1.*Numerator) / (Denominator**2) * dProb
  dScores += np.exp(Scores) * dDenominator
  tmp = np.zeros(Scores.shape)
  tmp[y,np.arange(num_train)] = 1.
  dScores += tmp * np.exp(Scores) * dNumerator
  dW = dScores.dot(X.T)
  dW /= num_train
  dW += (reg) * W

  return loss, dW
