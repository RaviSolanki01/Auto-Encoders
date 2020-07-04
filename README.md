# Auto-Encoders
various implementations of auto encoders




```python
import tensorflow as tf 
```


```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
```


```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
```

![autoencoder](https://www.learnopencv.com/wp-content/uploads/2017/11/denoising-example.png)


```python
print(mnist.train.images.shape)
print(mnist.test.images.shape)
print(mnist.validation.images.shape)
```

    (55000, 784)
    (10000, 784)
    (5000, 784)



```python
learning_rate =0.001
num_steps = 500
batch_size = 128
display_step = 10

num_input = 28
num_classes = 10
dropout = 0.75

x = tf.placeholder(tf.float32,[None,num_input,num_input,1])
#y = tf.placeholder(tf.float32,[None,num_classes])

#keep_prob = tf.placeholder(tf.float64)
```

![encoder](https://www.learnopencv.com/wp-content/uploads/2017/11/encoder-block-noise-2.png)


```python
def conv2d(x,W,b,strides=1):
    x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def maxpool2d(x,k=2):
    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

def conv2d_transpose(x, W, b, output_shape, strides=2):
    x = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)
```


```python
weights = {
    'encoder_h1': tf.get_variable('W0', shape=(3,3,1,32)),
    'encoder_h2': tf.get_variable('W1', shape=(3,3,32,32)),
    'decoder_h1': tf.get_variable('W2', shape=(3,3,32,32)),
    'decoder_h2': tf.get_variable('W3', shape=(3,3,32,32)),
    'decoder_h3': tf.get_variable('W4', shape=(3,3,32,32)),
    'decoder_h4': tf.get_variable('W5', shape=(3,3,32,1)),
}
biases = {
    'encoder_b1': tf.get_variable('B0', shape=(32)),
    'encoder_b2': tf.get_variable('B1', shape=(32)),
    'decoder_b1': tf.get_variable('B2', shape=(32)),
    'decoder_b2': tf.get_variable('B3', shape=(32)),
    'decoder_b3': tf.get_variable('B4', shape=(32)),
    'decoder_b4': tf.get_variable('B5', shape=(1)),
}
```


```python
def encoder(x):
    conv1 = conv2d(x,weights['encoder_h1'],biases['encoder_b1'])
    conv1 = maxpool2d(conv1)
    
    conv2 = conv2d(conv1,weights['encoder_h2'],biases['encoder_b2'])
    conv2 = maxpool2d(conv2)
    
    return conv2
```

![decoder](https://www.learnopencv.com/wp-content/uploads/2017/11/decoder-noise-diagram-3.png)


```python
def decoder(x):
    conv1 = conv2d(x,weights['decoder_h1'],biases['decoder_b1'])
    conv1 =  conv2d_transpose(conv1,weights['decoder_h2'],biases['decoder_b2'],output_shape=[1,14,14,32])
    
    conv2 = conv2d_transpose(conv1,weights['decoder_h3'],biases['decoder_b3'],output_shape=[1,28,28,32])
    conv2 = conv2d(conv2,weights['decoder_h4'],biases['decoder_b4'])
    
    return conv2
```


```python
encoder_op = encoder(x)
decoder_op = decoder(encoder_op)
```


```python
decoded = tf.sigmoid(decoder_op)
```


```python
targets_ = tf.placeholder(tf.float32,[None,28,28,1])
```


```python
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=decoder_op,labels=targets_)

learning_rate=tf.placeholder(tf.float32)
cost = tf.reduce_mean(loss)  #cost
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost) #optimizer
```


```python
init = tf.global_variables_initializer()
```


```python

```
