import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *
import pdb

def two_layer_convnet(X, model, y=None, reg=0.0,dropout=False,dropP=0.5):
  """
  Compute the loss and gradient for a simple two-layer ConvNet. The architecture
  is conv-relu-pool-affine-softmax, where the conv layer uses stride-1 "same"
  convolutions to preserve the input size; the pool layer uses non-overlapping
  2x2 pooling regions. We use L2 regularization on both the convolutional layer
  weights and the affine layer weights.

  Inputs:
  - X: Input data, of shape (N, C, H, W)
  - model: Dictionary mapping parameter names to parameters. A two-layer Convnet
    expects the model to have the following parameters:
    - W1, b1: Weights and biases for the convolutional layer
    - W2, b2: Weights and biases for the affine layer
  - y: Vector of labels of shape (N,). y[i] gives the label for the point X[i].
  - reg: Regularization strength.

  Returns:
  If y is None, then returns:
  - scores: Matrix of scores, where scores[i, c] is the classification score for
    the ith input and class c.

  If y is not None, then returns a tuple of:
  - loss: Scalar value giving the loss.
  - grads: Dictionary with the same keys as model, mapping parameter names to
    their gradients.
  """
  
  # Unpack weights
  W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, C, H, W = X.shape

  # We assume that the convolution is "same", so that the data has the same
  # height and width after performing the convolution. We can then use the
  # size of the filter to figure out the padding.
  conv_filter_height, conv_filter_width = W1.shape[2:]
  assert conv_filter_height == conv_filter_width, 'Conv filter must be square'
  assert conv_filter_height % 2 == 1, 'Conv filter height must be odd'
  assert conv_filter_width % 2 == 1, 'Conv filter width must be odd'
  conv_param = {'stride': 1, 'pad': (conv_filter_height - 1) / 2}
  pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

  # Compute the forward pass
  a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
  scores, cache2 = affine_forward(a1, W2, b2)

  if y is None:
    return scores

  # Compute the backward pass
  data_loss, dscores = softmax_loss(scores, y)

  # Compute the gradients using a backward pass
  da1, dW2, db2 = affine_backward(dscores, cache2)
  dX,  dW1, db1 = conv_relu_pool_backward(da1, cache1)

  # Add regularization
  dW1 += reg * W1
  dW2 += reg * W2
  reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2])

  loss = data_loss + reg_loss
  grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
  
  return loss, grads


def init_two_layer_convnet(weight_scale=1e-3, bias_scale=0, input_shape=(3, 32, 32),
                           num_classes=10, num_filters=32, filter_size=5):
  """
  Initialize the weights for a two-layer ConvNet.

  Inputs:
  - weight_scale: Scale at which weights are initialized. Default 1e-3.
  - bias_scale: Scale at which biases are initialized. Default is 0.
  - input_shape: Tuple giving the input shape to the network; default is
    (3, 32, 32) for CIFAR-10.
  - num_classes: The number of classes for this network. Default is 10
    (for CIFAR-10)
  - num_filters: The number of filters to use in the convolutional layer.
  - filter_size: The width and height for convolutional filters. We assume that
    all convolutions are "same", so we pick padding to ensure that data has the
    same height and width after convolution. This means that the filter size
    must be odd.

  Returns:
  A dictionary mapping parameter names to numpy arrays containing:
    - W1, b1: Weights and biases for the convolutional layer
    - W2, b2: Weights and biases for the fully-connected layer.
  """
  C, H, W = input_shape
  assert filter_size % 2 == 1, 'Filter size must be odd; got %d' % filter_size

  model = {}
  model['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
  model['b1'] = bias_scale * np.random.randn(num_filters)
  model['W2'] = weight_scale * np.random.randn(num_filters * H * W / 4, num_classes)
  model['b2'] = bias_scale * np.random.randn(num_classes)
  return model


def small_vgg_convnet(X, model, y=None, reg = 0.0, dropout = False,dropP=0.5):
    """
    Small VGG convet, the architecture looks like:
    [conv-relu-conv-relu-pool]xN - [affine]xM - [softmax or SVM]
    In this example, we pick N = 2, M = 2, softmax.
    This particular net is constructed as
    1{conv-relu}-2{conv-relu-pool}-3{conv-relu}-4{conv-relu-pool}-5{affine-relu}-6{affine}-{softmax}
    """
    # Unpack weights
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6 = (
            model['W1'], model['b1'], model['W2'], model['b2'],
            model['W3'], model['b3'], model['W4'], model['b4'],
            model['W5'], model['b5'], model['W6'], model['b6']) 
    N, C, H, W = X.shape
    
    conv_filter_height1, conv_filter_width1 = W1.shape[2:]
    assert conv_filter_height1 == conv_filter_height1, 'Conv filter must be square'
    assert conv_filter_height1 % 2 == 1, 'Conv filter height must be odd'
    assert conv_filter_width1 % 2 == 1, 'Conv filter width must be odd'
    
    conv_filter_height2, conv_filter_width2 = W2.shape[2:]
    assert conv_filter_height2 == conv_filter_height2, 'Conv filter must be square'
    assert conv_filter_height2 % 2 == 1, 'Conv filter height must be odd'
    assert conv_filter_width2 % 2 == 1, 'Conv filter width must be odd'
    
    conv_filter_height3, conv_filter_width3 = W3.shape[2:]
    assert conv_filter_height3 == conv_filter_height3, 'Conv filter must be square'
    assert conv_filter_height3 % 2 == 1, 'Conv filter height must be odd'
    assert conv_filter_width3 % 2 == 1, 'Conv filter width must be odd'
    
    conv_filter_height4, conv_filter_width4 = W4.shape[2:]
    assert conv_filter_height4 == conv_filter_height4, 'Conv filter must be square'
    assert conv_filter_height4 % 2 == 1, 'Conv filter height must be odd'
    assert conv_filter_width4 % 2 == 1, 'Conv filter width must be odd'
       
    conv_param = {'stride':1, 'pad': 1}
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    # Compute the forward pass
    # layer 1
    a1, cache1 = conv_relu_forward(X, W1, b1, conv_param)
    # layer 2
    a2, cache2 = conv_relu_pool_forward(a1, W2, b2, conv_param, pool_param)
    # layer 3
    a3, cache3 = conv_relu_forward(a2, W3, b3, conv_param)
    # layer 4
    a4, cache4 = conv_relu_pool_forward(a3, W4, b4, conv_param, pool_param)
    # layer 5
    a5, cache5 = affine_relu_forward(a4, W5, b5)
    # layer 6
    scores, cache6 = affine_forward(a5, W6, b6)

    if y is None:
        return scores
    
    # Compute the loss
    data_loss, dscores = softmax_loss(scores, y)

    # Compute the gradients using a backward pass
    # layer 6
    da5, dW6, db6 = affine_backward(dscores, cache6)
    # layer 5
    da4, dW5, db5 = affine_relu_backward(da5, cache5)
    # layer 4
    da3, dW4, db4 = conv_relu_pool_backward(da4, cache4)
    # layer 3
    da2, dW3, db3 = conv_relu_backward(da3, cache3)
    # layer2
    da1, dW2, db2 = conv_relu_pool_backward(da2, cache2)
    # layer1
    dX, dW1, db1 = conv_relu_backward(da1, cache1)

    # Add regularization
    dW1 += reg * W1
    dW2 += reg * W2
    dW3 += reg * W3
    dW4 += reg * W4
    dW5 += reg * W5
    dW6 += reg * W6
    reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2, W3, W4, W5, W6])

    loss = data_loss + reg_loss
    grads = {'W1': dW1, 'b1':db1,
             'W2': dW2, 'b2':db2,
             'W3': dW3, 'b3':db3,
             'W4': dW4, 'b4':db4,
             'W5': dW5, 'b5':db5,
             'W6': dW6, 'b6':db6
             }
    return loss, grads

def init_small_vgg_convnet(weight_scale = 1e-3, bias_scale=0, input_shape=(3, 32, 32),
        num_classes=10, num_filters_list=[64,64,128,128], filter_size=3,
        fc_size=1024):
    """
    Initialize the weights for the small vgg convnet
    Here is the size of activations and weights on each layer
    Input: [32x32x3], weights; 0
    1. conv-relu: [32x32x64], weights: (3*3*3)*64
    2. conv-relu: [32x32x64], weights: (3*3*64)*64
       pool: [16x16x64], weights: 0
    3. conv-relu: [16x16x128], weights: (3*3*64)*128
    4. conv-relu: [16x16x128], weights: (3*3*128)*128
       pool: [8x8x128], weights: 0.
    5. affine-relu: [1x1x(1024)], weights: (8*8*128)*(1024)
    6. affine-relu: [1x1x10], weights: (1024)*10
    """

    C, H, W = input_shape
    num_filters1 = num_filters_list[0]
    num_filters2 = num_filters_list[1]
    num_filters3 = num_filters_list[2]
    num_filters4 = num_filters_list[3]
    assert filter_size % 2 == 1, 'Filter size must be odd; got %d'% filter_size

    model = {}
    model['W1'] = weight_scale * np.random.randn(num_filters1, C, filter_size, filter_size)
    model['b1'] = bias_scale * np.random.randn(num_filters1)
    
    model['W2'] = weight_scale * np.random.randn(num_filters2, num_filters1, 
                                                 filter_size, filter_size)
    model['b2'] = bias_scale * np.random.randn(num_filters2)
    
    model['W3'] = weight_scale * np.random.randn(num_filters3, num_filters2, 
                                                 filter_size, filter_size)
    model['b3'] = bias_scale * np.random.randn(num_filters3)
    
    model['W4'] = weight_scale * np.random.randn(num_filters4, num_filters3, 
                                                 filter_size, filter_size)
    model['b4'] = bias_scale * np.random.randn(num_filters4)

    model['W5'] = weight_scale * np.random.randn(num_filters4*H*W/4/4, fc_size)
    model['b5'] = bias_scale * np.random.randn(fc_size)

    model['W6'] = weight_scale * np.random.randn(fc_size, num_classes)
    model['b6'] = bias_scale * np.random.randn(num_classes)

    return model


def tiny_vgg_convnet(X, model, y=None, reg = 0.0, dropout = False,dropP=0.5):
    """
    Tiny VGG convet, the architecture looks like:
    [conv-relu-conv-relu-pool]xN - [affine]xM - [softmax or SVM]
    In this example, we pick N = 2, M = 2, softmax.
    This particular net is constructed as
    1{conv-relu}-2{conv-relu-pool}-3{conv-relu}-4{conv-relu-pool}-5{affine-relu}-6{affine}-{softmax}
    """
    # Unpack weights
    W1, b1, W2, b2, W6, b6 = (
            model['W1'], model['b1'], model['W2'], model['b2'],
            model['W6'], model['b6']) 
    N, C, H, W = X.shape
    
    conv_filter_height1, conv_filter_width1 = W1.shape[2:]
    assert conv_filter_height1 == conv_filter_height1, 'Conv filter must be square'
    assert conv_filter_height1 % 2 == 1, 'Conv filter height must be odd'
    assert conv_filter_width1 % 2 == 1, 'Conv filter width must be odd'
    
    conv_filter_height2, conv_filter_width2 = W2.shape[2:]
    assert conv_filter_height2 == conv_filter_height2, 'Conv filter must be square'
    assert conv_filter_height2 % 2 == 1, 'Conv filter height must be odd'
    assert conv_filter_width2 % 2 == 1, 'Conv filter width must be odd'
    
    #conv_filter_height3, conv_filter_width3 = W3.shape[2:]
    #assert conv_filter_height3 == conv_filter_height3, 'Conv filter must be square'
    #assert conv_filter_height3 % 2 == 1, 'Conv filter height must be odd'
    #assert conv_filter_width3 % 2 == 1, 'Conv filter width must be odd'
    
    #conv_filter_height4, conv_filter_width4 = W4.shape[2:]
    #assert conv_filter_height4 == conv_filter_height4, 'Conv filter must be square'
    #assert conv_filter_height4 % 2 == 1, 'Conv filter height must be odd'
    #assert conv_filter_width4 % 2 == 1, 'Conv filter width must be odd'
       
    conv_param = {'stride':1, 'pad': 1}
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    # Compute the forward pass
    # layer 1
    a1, cache1 = conv_relu_forward(X, W1, b1, conv_param)
    # layer 2
    a2, cache2 = conv_relu_pool_forward(a1, W2, b2, conv_param, pool_param)
    # layer 3
    #a3, cache3 = conv_relu_forward(a2, W3, b3, conv_param)
    # layer 4
    #a4, cache4 = conv_relu_pool_forward(a3, W4, b4, conv_param, pool_param)
    # layer 5
    #a5, cache5 = affine_relu_forward(a4, W5, b5)
    # layer 6
    scores, cache6 = affine_forward(a2, W6, b6)

    if y is None:
        return scores
    
    # Compute the loss
    data_loss, dscores = softmax_loss(scores, y)

    # Compute the gradients using a backward pass
    # layer 6
    da2, dW6, db6 = affine_backward(dscores, cache6)
    # layer 5
    #da4, dW5, db5 = affine_relu_backward(da5, cache5)
    # layer 4
    #da3, dW4, db4 = conv_relu_pool_backward(da4, cache4)
    # layer 3
    #da2, dW3, db3 = conv_relu_backward(da3, cache3)
    # layer2
    da1, dW2, db2 = conv_relu_pool_backward(da2, cache2)
    # layer1
    dX, dW1, db1 = conv_relu_backward(da1, cache1)

    # Add regularization
    dW1 += reg * W1
    dW2 += reg * W2
    #dW3 += reg * W3
    #dW4 += reg * W4
    #dW5 += reg * W5
    dW6 += reg * W6
    reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2, W6])

    loss = data_loss + reg_loss
    grads = {'W1': dW1, 'b1':db1,
             'W2': dW2, 'b2':db2,
             'W6': dW6, 'b6':db6
             }
    return loss, grads

def init_tiny_vgg_convnet(weight_scale = 1e-3, bias_scale=0, input_shape=(3, 32, 32),
        num_classes=10, num_filters_list=[64,64,128,128], filter_size=3,
        fc_size=1024):
    """
    Initialize the weights for the small vgg convnet
    Here is the size of activations and weights on each layer
    Input: [32x32x3], weights; 0
    1. conv-relu: [32x32x64], weights: (3*3*3)*64
    2. conv-relu: [32x32x64], weights: (3*3*64)*64
       pool: [16x16x64], weights: 0
    3. conv-relu: [16x16x128], weights: (3*3*64)*128
    4. conv-relu: [16x16x128], weights: (3*3*128)*128
       pool: [8x8x128], weights: 0.
    5. affine-relu: [1x1x(1024)], weights: (8*8*128)*(1024)
    6. affine-relu: [1x1x10], weights: (1024)*10
    """

    C, H, W = input_shape
    num_filters1 = num_filters_list[0]
    num_filters2 = num_filters_list[1]
    num_filters3 = num_filters_list[2]
    num_filters4 = num_filters_list[3]
    assert filter_size % 2 == 1, 'Filter size must be odd; got %d'% filter_size

    model = {}
    model['W1'] = weight_scale * np.random.randn(num_filters1, C, filter_size, filter_size)
    model['b1'] = bias_scale * np.random.randn(num_filters1)
    
    model['W2'] = weight_scale * np.random.randn(num_filters2, num_filters1, 
                                                 filter_size, filter_size)
    model['b2'] = bias_scale * np.random.randn(num_filters2)
    
    #model['W3'] = weight_scale * np.random.randn(num_filters3, num_filters2, 
    #                                             filter_size, filter_size)
    #model['b3'] = bias_scale * np.random.randn(num_filters3)
    
    #model['W4'] = weight_scale * np.random.randn(num_filters4, num_filters3, 
    #                                             filter_size, filter_size)
    #model['b4'] = bias_scale * np.random.randn(num_filters4)

    #model['W5'] = weight_scale * np.random.randn(num_filters4*H*W/4/4, fc_size)
    #model['b5'] = bias_scale * np.random.randn(fc_size)

    model['W6'] = weight_scale * np.random.randn(num_filters2*H*W/2/2, num_classes)
    model['b6'] = bias_scale * np.random.randn(num_classes)

    return model

   
def tiny2_vgg_convnet(X, model, y=None, reg = 0.0, dropout = False,dropP=0.5):
    """
    Small VGG convet, the architecture looks like:
    [conv-relu-conv-relu-pool]xN - [affine]xM - [softmax or SVM]
    In this example, we pick N = 2, M = 2, softmax.
    This particular net is constructed as
    1{conv-relu}-2{conv-relu-pool}-3{conv-relu}-4{conv-relu-pool}-5{affine-relu}-6{affine}-{softmax}
    """
    # Unpack weights
    W1, b1, W2, b2, W3, b3, W4, b4, W6, b6 = (
            model['W1'], model['b1'], model['W2'], model['b2'],
            model['W3'], model['b3'], model['W4'], model['b4'],
            model['W6'], model['b6']) 
    N, C, H, W = X.shape
    
    conv_filter_height1, conv_filter_width1 = W1.shape[2:]
    assert conv_filter_height1 == conv_filter_height1, 'Conv filter must be square'
    assert conv_filter_height1 % 2 == 1, 'Conv filter height must be odd'
    assert conv_filter_width1 % 2 == 1, 'Conv filter width must be odd'
    
    conv_filter_height2, conv_filter_width2 = W2.shape[2:]
    assert conv_filter_height2 == conv_filter_height2, 'Conv filter must be square'
    assert conv_filter_height2 % 2 == 1, 'Conv filter height must be odd'
    assert conv_filter_width2 % 2 == 1, 'Conv filter width must be odd'
    
    conv_filter_height3, conv_filter_width3 = W3.shape[2:]
    assert conv_filter_height3 == conv_filter_height3, 'Conv filter must be square'
    assert conv_filter_height3 % 2 == 1, 'Conv filter height must be odd'
    assert conv_filter_width3 % 2 == 1, 'Conv filter width must be odd'
    
    conv_filter_height4, conv_filter_width4 = W4.shape[2:]
    assert conv_filter_height4 == conv_filter_height4, 'Conv filter must be square'
    assert conv_filter_height4 % 2 == 1, 'Conv filter height must be odd'
    assert conv_filter_width4 % 2 == 1, 'Conv filter width must be odd'
       
    conv_param = {'stride':1, 'pad': 1}
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    # Compute the forward pass
    # layer 1
    a1, cache1 = conv_relu_forward(X, W1, b1, conv_param)
    # layer 2
    a2, cache2 = conv_relu_pool_forward(a1, W2, b2, conv_param, pool_param)
    # layer 3
    a3, cache3 = conv_relu_forward(a2, W3, b3, conv_param)
    # layer 4
    a4, cache4 = conv_relu_pool_forward(a3, W4, b4, conv_param, pool_param)
    # layer 5
    #a5, cache5 = affine_relu_forward(a4, W5, b5)
    # layer 6
    scores, cache6 = affine_forward(a4, W6, b6)

    if y is None:
        return scores
    
    # Compute the loss
    data_loss, dscores = softmax_loss(scores, y)

    # Compute the gradients using a backward pass
    # layer 6
    da4, dW6, db6 = affine_backward(dscores, cache6)
    # layer 5
    #da4, dW5, db5 = affine_relu_backward(da5, cache5)
    # layer 4
    da3, dW4, db4 = conv_relu_pool_backward(da4, cache4)
    # layer 3
    da2, dW3, db3 = conv_relu_backward(da3, cache3)
    # layer2
    da1, dW2, db2 = conv_relu_pool_backward(da2, cache2)
    # layer1
    dX, dW1, db1 = conv_relu_backward(da1, cache1)

    # Add regularization
    dW1 += reg * W1
    dW2 += reg * W2
    dW3 += reg * W3
    dW4 += reg * W4
    #dW5 += reg * W5
    dW6 += reg * W6
    reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2, W3, W4, W6])

    loss = data_loss + reg_loss
    grads = {'W1': dW1, 'b1':db1,
             'W2': dW2, 'b2':db2,
             'W3': dW3, 'b3':db3,
             'W4': dW4, 'b4':db4,
             'W6': dW6, 'b6':db6
             }
    return loss, grads

def init_tiny2_vgg_convnet(weight_scale = 1e-3, bias_scale=0, input_shape=(3, 32, 32),
        num_classes=10, num_filters_list=[64,64,128,128], filter_size=3,
        fc_size=1024):
    """
    Initialize the weights for the small vgg convnet
    Here is the size of activations and weights on each layer
    Input: [32x32x3], weights; 0
    1. conv-relu: [32x32x64], weights: (3*3*3)*64
    2. conv-relu: [32x32x64], weights: (3*3*64)*64
       pool: [16x16x64], weights: 0
    3. conv-relu: [16x16x128], weights: (3*3*64)*128
    4. conv-relu: [16x16x128], weights: (3*3*128)*128
       pool: [8x8x128], weights: 0.
    5. affine-relu: [1x1x(1024)], weights: (8*8*128)*(1024)
    6. affine-relu: [1x1x10], weights: (1024)*10
    """

    C, H, W = input_shape
    num_filters1 = num_filters_list[0]
    num_filters2 = num_filters_list[1]
    num_filters3 = num_filters_list[2]
    num_filters4 = num_filters_list[3]
    assert filter_size % 2 == 1, 'Filter size must be odd; got %d'% filter_size

    model = {}
    model['W1'] = weight_scale * np.random.randn(num_filters1, C, filter_size, filter_size)
    model['b1'] = bias_scale * np.random.randn(num_filters1)
    
    model['W2'] = weight_scale * np.random.randn(num_filters2, num_filters1, 
                                                 filter_size, filter_size)
    model['b2'] = bias_scale * np.random.randn(num_filters2)
    
    model['W3'] = weight_scale * np.random.randn(num_filters3, num_filters2, 
                                                 filter_size, filter_size)
    model['b3'] = bias_scale * np.random.randn(num_filters3)
    
    model['W4'] = weight_scale * np.random.randn(num_filters4, num_filters3, 
                                                 filter_size, filter_size)
    model['b4'] = bias_scale * np.random.randn(num_filters4)

    #model['W5'] = weight_scale * np.random.randn(num_filters4*H*W/2/2, fc_size)
    #model['b5'] = bias_scale * np.random.randn(fc_size)

    model['W6'] = weight_scale * np.random.randn(num_filters4*H*W/4/4, num_classes)
    model['b6'] = bias_scale * np.random.randn(num_classes)

    return model














