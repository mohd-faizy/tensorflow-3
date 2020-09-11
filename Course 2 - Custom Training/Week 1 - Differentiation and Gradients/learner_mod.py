# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + colab={} colab_type="code" id="jqev488WJ9-R"
import tensorflow as tf
import numpy as np


# + [markdown] colab_type="text" id="sIIXMZboUw-P"
# # Exercise on basic Tensor operations

# + colab={} colab_type="code" id="MYdVyiSoLPgO"
# Convert NumPy array to Tensor using `tf.constant`
def tf_constant(array):
    
    ### START CODE HERE ###
    tf_constant_array = tf.constant(array)
    ### END CODE HERE ###
    return tf_constant_array


# + colab={} colab_type="code" id="W6BTwNJCLjV8"
# Square the input tensor x
def tf_square(array):
    
    ### START CODE HERE ###
    tf_squared_array = tf.square(array)
    ### END CODE HERE ###
    return tf_squared_array


# + colab={} colab_type="code" id="7nzBSX8-L0Xt"
# Reshape tensor x into a 3 x 3 matrix
def tf_reshape(array, shape):
    
    ### START CODE HERE ###
    tf_reshaped_array = tf.reshape(array, shape)
    ### END CODE HERE ###
    return tf_reshaped_array


# + colab={} colab_type="code" id="VoT-jiAIL8x5"
# Cast tensor x into float32 
def tf_cast(array, dtype):
    
    ### START CODE HERE ###
    tf_cast_array = tf.cast(array, dtype)
    ### END CODE HERE ###
    return tf_cast_array


# + colab={} colab_type="code" id="ivepGtD5MKP5"
# Multiply tensor x and y
def tf_multiply(num1, num2):
    
    ### START CODE HERE ###
    product = tf.multiply(num1, num2)
    ### END CODE HERE ###
    return product



# + colab={} colab_type="code" id="BVlntdYnMboh"
# Add tensor x and y
def tf_add(num1, num2):
    
    ### START CODE HERE ###
    sum = num1 + num2
    ### END CODE HERE ###
    return sum


# + [markdown] colab_type="text" id="9EN0W15EWNjD"
# # Exercise on Gradient Tape

# + colab={} colab_type="code" id="p3K94BWZM6nW"
def tf_gradient_tape(x):
    
    with tf.GradientTape() as t:
        
    ### START CODE HERE ###
        # Record the actions performed on tensor x with `watch`
        t.watch(x)		    

        # Define a polynomial of form 3x^3 - 2x^2 + x
        y =  3 * (x ** 3) - 2 * (x ** 2)  + x    

        # Obtain the sum of variable y
        z = tf.reduce_sum(y) 
  
    # Derivative of z wrt the original input tensor x
    dz_dx = t.gradient(z, x)
    ### END CODE HERE
    
    return dz_dx
