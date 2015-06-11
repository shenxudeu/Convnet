import numpy as np
from im2col import im2col_indices, col2im_indices
import copy
import pdb

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  N = x.shape[0]
  D = np.prod(x.shape) / N
  x_rows = x.reshape((N,D))
  out = np.dot(x_rows, w) + b
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  N = x.shape[0]
  D = np.prod(x.shape) / N
  x_rows = x.reshape((N,D))
  dw = np.dot(x_rows.T, dout)
  db = np.sum(dout, axis=0,keepdims=False)
  dx = np.dot(dout, w.T)
  dx = dx.reshape(x.shape)
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
 
  out = np.maximum(0,x)
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  dx = dout
  dx[x <= 0] = 0

  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width WW.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)

  """

  N,C,H,W = x.shape 
  F,C,HH,WW = w.shape
  stride, pad = conv_param['stride'], conv_param['pad']
  H_prime = 1. + float(H + 2 * pad - HH) / float(stride)
  W_prime = 1. + float(W + 2 * pad - WW) / float(stride)
  assert H_prime % 1 == 0
  assert W_prime % 1 == 0
  H_prime,W_prime = int(H_prime), int(W_prime)

  # pad input array
  x_padded = np.pad(x, ((0,0), (0,0), (pad, pad), (pad, pad)), 'constant')
  H_padded, W_padded = x_padded.shape[2], x_padded.shape[3]
  # naive implementation of im2col
  x_cols = None
  for i in xrange(HH, H_padded+1, stride):
    for j in xrange(WW, W_padded+1, stride):
      for n in xrange(N):
        field = x_padded[n,:,i-HH:i, j-WW:j].reshape((1,C*HH*WW))    
        if x_cols is None:
            x_cols = field
        else:
            x_cols = np.vstack((x_cols, field))
  
  # x_cols shape: (HH * WW * C) x (H_prime * W_prime * N)
  x_cols = x_cols.T

  #w2col, get w into shape of (F) x (HH * WW * C) 
  w_cols = w.reshape((F, C*HH *WW))
  
  
  # out_cols shape = (F) x (H_prime * W_prime * N)
  out_cols = np.dot(w_cols, x_cols) + b.reshape((b.shape[0],1))

  # out shape = N x F x H' x W'
  out = out_cols.reshape(F, H_prime, W_prime, N)
  out = out.transpose(3, 0, 1, 2) # (N, F, H', W')

  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives. (N, F, H', W')
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  
  """
  dx, dw, db = None, None, None
  x, w, b, conv_param = cache
  stride, pad = conv_param['stride'], conv_param['pad']
  N,C,H,W = x.shape 
  F,C,HH,WW = w.shape
 
  H_prime = 1. + float(H + 2 * pad - HH) / float(stride)
  W_prime = 1. + float(W + 2 * pad - WW) / float(stride)
  assert H_prime % 1 == 0
  assert W_prime % 1 == 0
  H_prime,W_prime = int(H_prime), int(W_prime)

  db = np.sum(dout, (0, 2, 3)) # sum along axis N, H', and W'
  
  # pad input array
  x_padded = np.pad(x, ((0,0), (0,0), (pad, pad), (pad, pad)), 'constant')
  H_padded, W_padded = x_padded.shape[2], x_padded.shape[3]
  # naive implementation of im2col
  x_cols = None
  for i in xrange(HH, H_padded+1, stride):
    for j in xrange(WW, W_padded+1, stride):
      for n in xrange(N):
        field = x_padded[n,:,i-HH:i, j-WW:j].reshape((1,C*HH*WW))    
        if x_cols is None:
            x_cols = field
        else:
            x_cols = np.vstack((x_cols, field))
  # x_cols shape: (HH * WW * C) x (H' * W' * N)
  x_cols = x_cols.T
  
  dout_ = dout.transpose(1, 2, 3, 0) # (F, H', W', N)
  dout_cols = dout_.reshape(F, H_prime * W_prime * N) # (F) x (H' * W' * N)

  dw_cols = np.dot(dout_cols, x_cols.T) # (F) x (HH * WW * C) 
  dw = dw_cols.reshape(F, C, HH, WW) # (F, C, HH, WW)

  w_cols = w.reshape(F, C*HH*WW) # (F) x (HH * WW * C)
  dx_cols = np.dot(w_cols.T, dout_cols) # (HH * WW * C) x (H' * W' * N)
  
  # col2im: convert back from (d)x_cols to (d)x
  #dx = col2im_indices(dx_cols, (N, C, H, W), HH, WW, pad, stride)
  #dx_cols = dx_cols.T # (H' * W' * N) x (HH * WW * C)
  dx_padded = np.zeros((N, C, H_padded, W_padded))
  idx = 0
  for i in xrange(HH, H_padded+1, stride):
    for j in xrange(WW, W_padded+1, stride):
      for n in xrange(N):
        dx_padded[n:n+1,:,i-HH:i,j-WW:j] += dx_cols[:,idx].reshape(1,C,HH,WW)
        idx += 1
  dx = dx_padded[:,:,pad:-pad,pad:-pad]
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)

  """
  N, C, H, W = x.shape
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
  assert (H - pool_height) % stride == 0
  assert (W - pool_width) % stride == 0
  out_height = (H - pool_height) / stride + 1
  out_width = (W - pool_width) / stride + 1
  
  out = np.zeros((N,C, out_height, out_width))
  for c in xrange(C):
      for n in xrange(N):
          idx_i = 0
          for i in xrange(pool_height, H+1, stride):
              idx_j = 0
              for j in xrange(pool_width, W+1, stride):
                  field = x[n,c,i-pool_height:i,j-pool_width:j]
                  out[n,c,idx_i, idx_j] = np.max(field)
                  idx_j += 1
              idx_i += 1
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  x, pool_param = cache
  dx = np.zeros(x.shape)
  
  N, C, H, W = x.shape
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
  assert (H - pool_height) % stride == 0
  assert (W - pool_width) % stride == 0
  
  for c in xrange(C):
      for n in xrange(N):
          idx_i = 0
          for i in xrange(pool_height, H+1, stride):
              idx_j = 0
              for j in xrange(pool_width, W+1, stride):
                  field = x[n,c,i-pool_height:i,j-pool_width:j]
                  field_col = np.zeros((1, pool_height*pool_width))
                  field_col[0,np.argmax(field.reshape((1,-1)))] = 1.
                  
                  field_col *= dout[n,c,idx_i,idx_j]
                  dx[n,c,i-pool_height:i,j-pool_width:j] += field_col.reshape(field.shape)
                  idx_j += 1
              idx_i += 1
 
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

