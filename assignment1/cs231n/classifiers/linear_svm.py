import numpy as np
from random import shuffle
import pdb

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    # forward
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    dscores = np.zeros(scores.shape)
    dcorrect_class_score = 0.
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
      # backward
      dloss = (1.)
      if margin > 0:
        dmargin = (1.) * dloss
      else:
        dmargin = (0.)
      dscores[j] = (1.) * dmargin
      dcorrect_class_score += (-1.) * dmargin
    dscores[y[i]] = (1.) * dcorrect_class_score
    dW += dscores.reshape(len(dscores),1).dot(X[:,i].reshape(len(X[:,i]),1).T)
  
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += (reg) * W
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  # forward
  Score = W.dot(X)
  Scorey = np.tile(Score[y,np.arange(num_train)].reshape((1,num_train)),(num_classes,1))
  margin0 = Score-Scorey + 1.
  Margin = np.maximum(margin0,0)
  Margin[y,np.arange(num_train)] = 0.
  loss = np.sum(Margin)
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)

  # backward
  dScore = np.zeros(Score.shape)
  dloss = 1.
  dMargin = np.ones(Margin.shape)*dloss
  tmp = np.zeros(margin0.shape)
  tmp[margin0>0.] = 1.
  dmargin0 = (tmp) * dMargin
  dScore = (1.) * dmargin0
  dScorey = (-1.) * dmargin0
  
  tmp = np.zeros(Score.shape)
  tmp[y,np.arange(num_train)] = 1.
  counts = np.zeros(margin0.shape)
  counts[margin0>0] = 1.
  counts = np.sum(counts,axis=0)
  dScore += (tmp) *dScorey * (counts)
  dW = dScore.dot(X.T)
  dW /= num_train
  dW += (reg) * W

  return loss, dW
