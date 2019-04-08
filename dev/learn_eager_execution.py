#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : learn_eager_execution.py
# @Author: harry
# @Date  : 2019/4/7 下午10:52
# @Desc  : Just a simple toy to learn tf eager execution basics

import tensorflow as tf
import numpy as np
import time
import tempfile

tf.enable_eager_execution()

# Tensors
# using python native types
print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3, 4, 5]))
print(tf.encode_base64("hello world"))
# operator overloading
print(tf.square(2) + tf.square(3))

x = tf.matmul([[1]], [[2, 3]])
print(x)
print(x.shape)
print(x.dtype)

# Numpy compatibility
ndarray = np.ones([3, 3])
# TensorFlow operations convert numpy arrays to Tensors automatically
# TensorFlow will perform GPU-CPU memory copy automatically
tensor = tf.multiply(ndarray, 42)
print(tensor)
# And NumPy operations convert Tensors to numpy arrays automatically
print(np.add(tensor, 1))
# The .numpy() method explicitly converts a Tensor to a numpy array
print(tensor.numpy())

# GPU acceleration
x = tf.random_uniform([3, 3])
# Is there a GPU available
print(tf.test.is_gpu_available())
# Is the Tensor on GPU #0
print(x.device)
print(x.device.endswith('GPU:0'))


# Explicit device placement
def time_matmul(x):
    start = time.time()
    exec_times = 50
    for _ in range(exec_times):
        tf.matmul(x, x)

    result = time.time() - start
    print("{} loops: {:0.2f}ms".format(exec_times, 1000 * result))


# force exec on CPU
print("On CPU")
with tf.device("CPU:0"):
    x = tf.random_uniform([1000, 1000])
    assert x.device.endswith("CPU:0")
    time_matmul(x)

# force exec on GPU if possible
print("On GPU")
if tf.test.is_gpu_available():
    with tf.device("GPU:0"):
        x = tf.random_uniform([1000, 1000])
        assert x.device.endswith("GPU:0")
        time_matmul(x)

# Create a source Dataset
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
_, filename = tempfile.mkstemp()
with open(filename, 'w') as f:
    f.write("""Line1
Line2
Line3
    """)
ds_file = tf.data.TextLineDataset(filename)
# apply transformation to Dataset
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)
# now iterate over Dataset
print('Elements of ds_tensors:')
for x in ds_tensors:
    print(x)
print('\nElements in ds_file:')
for x in ds_file:
    print(x)
