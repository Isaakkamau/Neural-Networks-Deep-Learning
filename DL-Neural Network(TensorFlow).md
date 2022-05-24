# Neural Networks/Deep Learning


# <font color=red>_"A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."Tom Mitchell (1999)_</font>

- E: Experience (the number of times).
- T: The Task (driving a car).
- P: The Performance (good or bad).

## Deep learning is divided in 5 major parts
1. Perceptrons
2. Recognition
3. Training
4. Testing
5. Learning

<img src="https://www.smartsheet.com/sites/default/files/IC-simplified-artificial-neural-networks-corrected.svg" alt="Neural Networks image" title="Neural Networks image" />

# Deep learning / Neural Networks

>Neurons (aka Nerve Cells) are the fundamental units of human brain and nervous system.
They receive input from the external world, for sending output (commands to our muscles), and for transforming the electrical signals in between.
##### Artificial Neural Networks 
>Neural networks are in fact multi-layer Perceptrons(The perceptron defines the first step into multi-layered neural networks.)

## Example of Neural Networks
- ### TensorFlow Playground
>With the below TensorFlow Playground you can learn about Neural Networks (NN)


```python
%%html
<iframe src="https://playground.tensorflow.org" width="1200" height="1000"></iframe>
```


<iframe src="https://playground.tensorflow.org" width="1200" height="1000"></iframe>



 

 

 

 

 

# CHAPTER 1. <font color=red>Perceptron</font>
>A perceptron is a neural network unit (an artificial neuron) that does certain computations to detect features or business intelligence in the input data.

 

 

 

<img src="https://indiantechwarrior.com/wp-content/uploads/2021/04/perceptron-1.png" alt="Perceptron image" title="Perceptrons input and output" />

>The idea was to use different weights to represent the importance of each input, and that the sum of the values should be greater than a threshold value before making a decision like true or false (0 or 1).

### Perceptron Example

Imagine a perceptron (in your brain).

The perceptron tries to decide if you should go to a concert.

- Is the artist good? Is the weather good?

- What weights should these facts have?

|Criteria         |Input        |Weight    |
|-----------------|-------------|----------|
|Artists is Good  |x1 = 0 or 1  |w1 = 0.7  |
|Weather is Good  |	x2 = 0 or 1 |w2 = 0.6  |
|Friend will Come |x3 = 0 or 1  |w3 = 0.5  |
|Food is Served   |x4 = 0 or 1l |w4 = 0.3  |

### The Perceptron Algorithm

Frank Rosenblatt suggested this algorithm:

    Set a threshold value
    Multiply all inputs with its weights
    Sum all the results
    Activate the output


### 1. Set a threshold value:

    Threshold = 1.5

### 2. Multiply all inputs with its weights:

    x1 * w1 = 1 * 0.7 = 0.7
    x2 * w2 = 0 * 0.6 = 0
    x3 * w3 = 1 * 0.5 = 0.5
    x4 * w4 = 0 * 0.3 = 0
    x5 * w5 = 1 * 0.4 = 0.4

### 3. Sum all the results:

    0.7 + 0 + 0.5 + 0 + 0.4 = 1.6 (The Weighted Sum)

### 4. Activate the Output:

    Return true if the sum > 1.5 ("Yes I will go to the Concert")



### <font color=red>Note</font>

>If the weather weight is 0.6 for you, it might different for someone else. A higher weight means that the weater is more important to them.

>If the treshold value is 1.5 for you, it might be different for someone else. A lower treshold means they are more wanting to go to the consert.


### Perceptron Terminology

    Perceptron Inputs
    Node values
    Node Weights
    Activation Function



### Perceptron Inputs

   >Perceptron inputs are called nodes.
   The nodes have both a <font color=red>value</font> and <font color=red>a weight.</font>

### Node Values

>In the example above, the node values are: 1, 0, 1, 0, 1

>The binary input values (0 or 1) can be interpreted as (no or yes) or (false or true).
### Node Weights

>Weights shows the strength of each node.

>In the example above, the node weights are: 0.7, 0.6, 0.5, 0.3, 0.4
### The Activation Function

>The activation functions maps the result (the weighted sum) into a required value like 0 or 1.

>In the example above, the activation function is simple: (sum > 1.5)

>The binary output (1 or 0) can be interpreted as (yes or no) or (true or false).


### <font color=red>Note</font>

    It is obvious that a decision is NOT made by one neuron alone.

    Other neurons must provide input: Is the artist good. Is the weather good...

      In Neuroscience, there is a debate if single-neuron encoding or distributed encoding is most relevant for understanding brain functions.


# Neural Networks
### The Perceptron defines the first step into Neural Networks.

### Multi-Layer Perceptrons can be used for very sophisticated decision making.

<img src="https://www.w3schools.com/ai/img_neural_networks.jpg" alt="Neural Networks image" title="Multi-Layer Perceptrons" />

In the __Neural Network Model__, input data (yellow) are processed against a hidden layer (blue) and modified against more hidden layers (green) to produce the final output (red).

#### The First Layer:
The 3 yellow perceptrons are making 3 simple decisions based on the input evidence. Each single desision is sent to the 4 perceptrons in the next layer.

#### The Second Layer:
The blue perceptrons are making decisions by weighing the results from the first layer. This layer make more complex decisions at a more abstract level than the first layer.

#### The Third Layer:
Even more complex decisions are made by the green perceptons.

 

 

 

 

# CHAPTER 2. <font color=red>Pattern Recognition</font>

 

 

 

Imagine a strait line (a linear graph) in a space with scattered x y points.
How can you classify the points over and under the line?


![index.png](attachment:index.png)

A perceptron can be trained to recognize the points over the line, without knowing the formula for the line.

Perceptron

![Screenshot%202022-05-16%20at%2017-00-34%20Pattern%20Recognition.png](attachment:Screenshot%202022-05-16%20at%2017-00-34%20Pattern%20Recognition.png)

### How to Program a Perceptron

To learn more about how to program a perceptron, we will create a very simple JavaScript program that will:

    Create a simple plotter
    Create 500 random x y points
    Diplay the x y points
    Create a line function: f(x)
    Display the line
    Compute the desired answers
    Display the desired answers



```python
<!DOCTYPE html>
<html>
<script src="myplotlib.js"></script>

<body>
<canvas id="myCanvas" width="400px" height="400px" style="width:100%;max-width:400px;border:1px solid black"></canvas>

<script>
// Create a Plotter
const plotter = new XYPlotter("myCanvas");
plotter.transformXY();
const xMax = plotter.xMax;
const yMax = plotter.yMax;
const xMin = plotter.xMin;
const yMin = plotter.yMin;

// Create Random XY Points
const numPoints = 500;
const xPoints = [];
const yPoints = [];
for (let i = 0; i < numPoints; i++) {
  xPoints[i] = Math.random() * xMax;
  yPoints[i] = Math.random() * yMax;
}

// Line Function
function f(x) {
  return x * 1.2 + 50;
}

//Plot the Line
plotter.plotLine(xMin, f(xMin), xMax, f(xMax), "black");

// Compute Desired Answers
const desired = [];
for (let i = 0; i < numPoints; i++) {
  desired[i] = 0;
  if (yPoints[i] > f(xPoints[i])) {desired[i] = 1}
}

// Diplay Desired Result
for (let i = 0; i < numPoints; i++) {
  let color = "blue";
  if (desired[i]) color = "black";
  plotter.plotPoint(xPoints[i], yPoints[i], color);
}
</script>
</body>
</html>

```


      File "C:\Users\Isaac\AppData\Local\Temp/ipykernel_9704/2427128494.py", line 1
        <!DOCTYPE html>
        ^
    SyntaxError: invalid syntax
    


![Screenshot%202022-05-16%20at%2017-05-18%20W3Schools%20online%20HTML%20editor.png](attachment:Screenshot%202022-05-16%20at%2017-05-18%20W3Schools%20online%20HTML%20editor.png)

perceptron accept two parameters:

    The number of inputs (no)
    The learning rate (learningRate).

Set the default learning rate to 0.00001.

Then create random weights between -1 and 1 for each input.

### Jargons used in creating a perceptron

#### The Random Weights

The Perceptron will start with a random weight for each input.

#### The Learning Rate

For each mistake, while training the Perceptron, the weights will be ajusted with a small fraction.

This small fraction is the "Perceptron's learning rate".

In the Perceptron object we call it learnc.
#### The Bias

Sometimes, if both inputs are zero, the perceptron might produce an in correct output.

To avoid this, we give the perceptron an extra input with the value of 1.

This is called a bias.

### Add an Activate Function

##### Remember the perceptron algorithm:

    Multiply each input with the perceptron's weights
    Sum the results
    Compute the outcome




## Learning is Looping

An ML model is Trained by Looping over data multiple times.

For each iteration, the Weight Values are adjusted.

Training is complete when the iterations fails to Reduce the Cost.


### A Trainer Object

Create a Trainer object that can take any number of (x,y) values in two arrays (xArr,yArr).

Set both weight and bias to zero.

A learning constant (learnc) has to be set, and a cost variable must be defined:

### Cost Function

A standard way to solve a regression problem, is with an "Cost Function" that measures how good the solution is.

The function uses the weight and bias from the model (y = wx + b) and returns an error, based on how well the line fits a plot.

The way to compute this error, is to loop through all (x,y) points in the plot, and sum the square distances between the y value of each point and the line.

The most conventional way is to square the distances (to ensure positive values) and to make the error function differentiable.



#### Another name for the Cost Function is Error Function.

The formula used in the function is actually this:
Formula
![Screenshot%202022-05-16%20at%2017-22-14%20Machine%20Learning.png](attachment:Screenshot%202022-05-16%20at%2017-22-14%20Machine%20Learning.png)

    E is the error (cost)
    N is the total number of observations (points)
    y is the value (label) of each observation
    x is the value (feature) of each observation
    m is the slope (weight)
    b is intercept (bias)
    mx + b is the prediction
    1/N * N∑1 is the squared mean value



### The Train Function

We will now run a gradient descent.

The gradient descent algorithm should walk the cost function towards the best line.

Each iteration should update both m and b towards a line with a lower cost (error).

To do that, we add a train function that loops over all the data many times:

#### An Update Weights Function

The train function above should update the weights and biases in each iteration.

The direction to move is calculated using two partial derivatives:

# Terminologies
## Relationships

Machine learning systems uses Relationships between Inputs to produce Predictions.

In algebra, a relationship is often written as y = ax + b:

    y is the label we want to predict
    a is the slope of the line
    x are the input values
    b is the intercept

With ML, a relationship is written as y = b + wx:

    y is the label we want to predict
    w is the weight (the slope)
    x are the features (input values)
    b is the intercept

Machine Learning Labels

In Machine Learning terminology, the label is the thing we want to predict.

It is like the y in a linear graph:
![Screenshot%202022-05-16%20at%2017-25-46%20Machine%20Learning%20Terminology.png](attachment:Screenshot%202022-05-16%20at%2017-25-46%20Machine%20Learning%20Terminology.png)

## Machine Learning Features

In Machine Learning terminology, the features are the input.

They are like the x values in a linear graph:



![Screenshot%202022-05-16%20at%2017-26-39%20Machine%20Learning%20Terminology.png](attachment:Screenshot%202022-05-16%20at%2017-26-39%20Machine%20Learning%20Terminology.png)
Sometimes there can be many features (input values) with different weights:

y = b + w1x1 + w2x2 + w3x3 + w4x4

## Machine Learning Models

A Model defines the relationship between the label (y) and the features (x).

There are three phases in the life of a model:

    Data Collection
    Training
    Inference 

## Machine Learning Training

The goal of training is to create a model that can answer a question. Like what is the expected price for a house?
Machine Learning Inference

Inference is when the trained model is used to infer (predict) values using live data. Like putting the model into production.


 

 

 

# TENSORFLOW


```python
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

 

 

 

 ### Install TensorFlow 2 
 ##### Requires the latest pip
pip install --upgrade pip

##### Current stable release for CPU and GPU
pip install tensorflow

##### Or try the preview build (unstable)
pip install tf-nightly



```python
pip install tensorflow
```

    Note: you may need to restart the kernel to use updated packages.Collecting tensorflow
    
      Downloading tensorflow-2.8.0-cp39-cp39-win_amd64.whl (438.0 MB)
    Collecting termcolor>=1.1.0
      Downloading termcolor-1.1.0.tar.gz (3.9 kB)
    Requirement already satisfied: wrapt>=1.11.0 in c:\users\isaac\anaconda3\lib\site-packages (from tensorflow) (1.12.1)
    Collecting protobuf>=3.9.2
      Downloading protobuf-3.20.1-cp39-cp39-win_amd64.whl (904 kB)
    Collecting grpcio<2.0,>=1.24.3
      Downloading grpcio-1.46.1-cp39-cp39-win_amd64.whl (3.5 MB)
    Collecting tf-estimator-nightly==2.8.0.dev2021122109
      Downloading tf_estimator_nightly-2.8.0.dev2021122109-py2.py3-none-any.whl (462 kB)
    Requirement already satisfied: typing-extensions>=3.6.6 in c:\users\isaac\anaconda3\lib\site-packages (from tensorflow) (3.10.0.2)
    Collecting google-pasta>=0.1.1
      Downloading google_pasta-0.2.0-py3-none-any.whl (57 kB)
    Collecting opt-einsum>=2.3.2
      Downloading opt_einsum-3.3.0-py3-none-any.whl (65 kB)
    Collecting gast>=0.2.1
      Downloading gast-0.5.3-py3-none-any.whl (19 kB)
    Collecting absl-py>=0.4.0
      Downloading absl_py-1.0.0-py3-none-any.whl (126 kB)
    Requirement already satisfied: setuptools in c:\users\isaac\anaconda3\lib\site-packages (from tensorflow) (58.0.4)
    Collecting keras-preprocessing>=1.1.1
      Downloading Keras_Preprocessing-1.1.2-py2.py3-none-any.whl (42 kB)
    Collecting libclang>=9.0.1
      Downloading libclang-14.0.1-py2.py3-none-win_amd64.whl (14.2 MB)
    Collecting astunparse>=1.6.0
      Downloading astunparse-1.6.3-py2.py3-none-any.whl (12 kB)
    Requirement already satisfied: numpy>=1.20 in c:\users\isaac\anaconda3\lib\site-packages (from tensorflow) (1.20.3)
    Collecting flatbuffers>=1.12
      Downloading flatbuffers-2.0-py2.py3-none-any.whl (26 kB)
    Requirement already satisfied: h5py>=2.9.0 in c:\users\isaac\anaconda3\lib\site-packages (from tensorflow) (3.2.1)
    Requirement already satisfied: six>=1.12.0 in c:\users\isaac\anaconda3\lib\site-packages (from tensorflow) (1.16.0)
    Collecting keras<2.9,>=2.8.0rc0
      Downloading keras-2.8.0-py2.py3-none-any.whl (1.4 MB)
    Collecting tensorboard<2.9,>=2.8
      Downloading tensorboard-2.8.0-py3-none-any.whl (5.8 MB)
    Collecting tensorflow-io-gcs-filesystem>=0.23.1
      Downloading tensorflow_io_gcs_filesystem-0.25.0-cp39-cp39-win_amd64.whl (1.5 MB)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in c:\users\isaac\anaconda3\lib\site-packages (from astunparse>=1.6.0->tensorflow) (0.37.0)
    Collecting tensorboard-plugin-wit>=1.6.0
      Downloading tensorboard_plugin_wit-1.8.1-py3-none-any.whl (781 kB)
    Collecting google-auth<3,>=1.6.3
      Downloading google_auth-2.6.6-py2.py3-none-any.whl (156 kB)
    Collecting google-auth-oauthlib<0.5,>=0.4.1
      Downloading google_auth_oauthlib-0.4.6-py2.py3-none-any.whl (18 kB)
    Collecting markdown>=2.6.8
      Downloading Markdown-3.3.7-py3-none-any.whl (97 kB)
    Requirement already satisfied: requests<3,>=2.21.0 in c:\users\isaac\anaconda3\lib\site-packages (from tensorboard<2.9,>=2.8->tensorflow) (2.26.0)
    Requirement already satisfied: werkzeug>=0.11.15 in c:\users\isaac\anaconda3\lib\site-packages (from tensorboard<2.9,>=2.8->tensorflow) (2.0.2)
    Collecting tensorboard-data-server<0.7.0,>=0.6.0
      Downloading tensorboard_data_server-0.6.1-py3-none-any.whl (2.4 kB)
    Collecting rsa<5,>=3.1.4
      Downloading rsa-4.8-py3-none-any.whl (39 kB)
    Collecting cachetools<6.0,>=2.0.0
      Downloading cachetools-5.1.0-py3-none-any.whl (9.2 kB)
    Collecting pyasn1-modules>=0.2.1
      Downloading pyasn1_modules-0.2.8-py2.py3-none-any.whl (155 kB)
    Collecting requests-oauthlib>=0.7.0
      Downloading requests_oauthlib-1.3.1-py2.py3-none-any.whl (23 kB)
    Requirement already satisfied: importlib-metadata>=4.4 in c:\users\isaac\anaconda3\lib\site-packages (from markdown>=2.6.8->tensorboard<2.9,>=2.8->tensorflow) (4.8.1)
    Requirement already satisfied: zipp>=0.5 in c:\users\isaac\anaconda3\lib\site-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<2.9,>=2.8->tensorflow) (3.6.0)
    Collecting pyasn1<0.5.0,>=0.4.6
      Downloading pyasn1-0.4.8-py2.py3-none-any.whl (77 kB)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\isaac\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.9,>=2.8->tensorflow) (2021.10.8)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\users\isaac\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.9,>=2.8->tensorflow) (1.26.7)
    Requirement already satisfied: charset-normalizer~=2.0.0 in c:\users\isaac\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.9,>=2.8->tensorflow) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\isaac\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.9,>=2.8->tensorflow) (3.2)
    Collecting oauthlib>=3.0.0
      Downloading oauthlib-3.2.0-py3-none-any.whl (151 kB)
    Building wheels for collected packages: termcolor
      Building wheel for termcolor (setup.py): started
      Building wheel for termcolor (setup.py): finished with status 'done'
      Created wheel for termcolor: filename=termcolor-1.1.0-py3-none-any.whl size=4847 sha256=4722b958a0aeefd4b8117a7e14c7541579215005c7fb45306e38884236eccb20
      Stored in directory: c:\users\isaac\appdata\local\pip\cache\wheels\b6\0d\90\0d1bbd99855f99cb2f6c2e5ff96f8023fad8ec367695f7d72d
    Successfully built termcolor
    Installing collected packages: pyasn1, rsa, pyasn1-modules, oauthlib, cachetools, requests-oauthlib, google-auth, tensorboard-plugin-wit, tensorboard-data-server, protobuf, markdown, grpcio, google-auth-oauthlib, absl-py, tf-estimator-nightly, termcolor, tensorflow-io-gcs-filesystem, tensorboard, opt-einsum, libclang, keras-preprocessing, keras, google-pasta, gast, flatbuffers, astunparse, tensorflow
    Successfully installed absl-py-1.0.0 astunparse-1.6.3 cachetools-5.1.0 flatbuffers-2.0 gast-0.5.3 google-auth-2.6.6 google-auth-oauthlib-0.4.6 google-pasta-0.2.0 grpcio-1.46.1 keras-2.8.0 keras-preprocessing-1.1.2 libclang-14.0.1 markdown-3.3.7 oauthlib-3.2.0 opt-einsum-3.3.0 protobuf-3.20.1 pyasn1-0.4.8 pyasn1-modules-0.2.8 requests-oauthlib-1.3.1 rsa-4.8 tensorboard-2.8.0 tensorboard-data-server-0.6.1 tensorboard-plugin-wit-1.8.1 tensorflow-2.8.0 tensorflow-io-gcs-filesystem-0.25.0 termcolor-1.1.0 tf-estimator-nightly-2.8.0.dev2021122109
    

# TensorFlow basics

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/guide/basics"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/basics.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/guide/basics.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
  <td>
    <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/guide/basics.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>
</table>

This guide provides a quick overview of TensorFlow basics. Each section of this doc is an overview of a larger topic—you can find links to full guides at the end of each section.

TensorFlow is an end-to-end platform for machine learning. It supports the following:

   - Multidimensional-array based numeric computation (similar to NumPy.)
   - GPU and distributed processing
   - Automatic differentiation
   - Model construction, training, and export
   - And more



## Tensors

TensorFlow operates on multidimensional arrays or _tensors_ represented as `tf.Tensor` objects. Here is a two-dimensional tensor:


```python
import tensorflow as tf

x = tf.constant([[1., 2., 3.],
                 [4., 5., 6.]])

print(x)
print(x.shape)
print(x.dtype)
```

    tf.Tensor(
    [[1. 2. 3.]
     [4. 5. 6.]], shape=(2, 3), dtype=float32)
    (2, 3)
    <dtype: 'float32'>
    

The most important attributes of a `tf.Tensor` are its `shape` and `dtype`:

* `Tensor.shape`: tells you the size of the tensor along each of its axes.
* `Tensor.dtype`: tells you the type of all the elements in the tensor.

TensorFlow implements standard mathematical operations on tensors, as well as many operations specialized for machine learning.

For example:


```python
x + x
```




    <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
    array([[ 2.,  4.,  6.],
           [ 8., 10., 12.]], dtype=float32)>




```python
5 * x
```




    <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
    array([[ 5., 10., 15.],
           [20., 25., 30.]], dtype=float32)>




```python
x @ tf.transpose(x)
```




    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[14., 32.],
           [32., 77.]], dtype=float32)>




```python
tf.concat([x, x, x], axis=0)
```




    <tf.Tensor: shape=(6, 3), dtype=float32, numpy=
    array([[1., 2., 3.],
           [4., 5., 6.],
           [1., 2., 3.],
           [4., 5., 6.],
           [1., 2., 3.],
           [4., 5., 6.]], dtype=float32)>




```python
tf.nn.softmax(x, axis=-1)
```




    <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
    array([[0.09003057, 0.24472848, 0.66524094],
           [0.09003057, 0.24472848, 0.66524094]], dtype=float32)>




```python
tf.reduce_sum(x)
```




    <tf.Tensor: shape=(), dtype=float32, numpy=21.0>



Running large calculations on CPU can be slow. When properly configured, TensorFlow can use accelerator hardware like GPUs to execute operations very quickly.


```python
if tf.config.list_physical_devices('CPU'):
  print("TensorFlow **IS** using the CPU")
else:
  print("TensorFlow **IS NOT** using the CPU")
```

    TensorFlow **IS** using the CPU
    


```python
if tf.config.list_physical_devices('GPU'):
  print("TensorFlow **IS** using the GPU")
else:
  print("TensorFlow **IS NOT** using the GPU")
```

    TensorFlow **IS NOT** using the GPU
    

# Variables

Normal `tf.Tensor` objects are immutable. To store model weights (or other mutable state) in TensorFlow use a `tf.Variable`.


```python
var = tf.Variable([0.0, 0.0, 0.0])
```


```python
var.assign([1, 2, 3])
```




    <tf.Variable 'UnreadVariable' shape=(3,) dtype=float32, numpy=array([1., 2., 3.], dtype=float32)>




```python
var.assign_add([1, 1, 1])
```




    <tf.Variable 'UnreadVariable' shape=(3,) dtype=float32, numpy=array([2., 3., 4.], dtype=float32)>



## Automatic differentiation

<a href="https://en.wikipedia.org/wiki/Gradient_descent" class="external">_Gradient descent_</a> and related algorithms are a cornerstone of modern machine learning.

To enable this, TensorFlow implements automatic differentiation (autodiff), which uses calculus to compute gradients. Typically you'll use this to calculate the gradient of a model's _error_ or _loss_ with respect to its weights.


```python
x = tf.Variable(1.0)

def f(x):
  y = x**2 + 2*x - 5
  return y
```


```python
f(x)
```




    <tf.Tensor: shape=(), dtype=float32, numpy=-2.0>



At `x = 1.0`, `y = f(x) = (1**2 + 2*1 - 5) = -2`.

The derivative of `y` is `y' = f'(x) = (2*x + 2) = 4`. TensorFlow can calculate this automatically:


```python
with tf.GradientTape() as tape:
  y = x**2 + 2*x - 5

g_x = tape.gradient(y, x)  # g(x) = dy/dx

g_x
```




    <tf.Tensor: shape=(), dtype=float32, numpy=4.0>



This simplified example only takes the derivative with respect to a single scalar (`x`), but TensorFlow can compute the gradient with respect to any number of non-scalar tensors simultaneously.

## Graphs and tf.function

While you can use TensorFlow interactively like any Python library, TensorFlow also provides tools for:

* **Performance optimization**: to speed up training and inference.
* **Export**: so you can save your model when it's done training.

These require that you use `tf.function` to separate your pure-TensorFlow code from Python.


```python
@tf.function
def my_func(x):
  print('Tracing...\n')
  return tf.reduce_sum(x)
```

The first time you run the `tf.function`, although it executes in Python, it captures a complete, optimized graph representing the TensorFlow computations done within the function.


```python
x = tf.constant([1, 2, 3])
my_func(x)
```

    Tracing...
    
    




    <tf.Tensor: shape=(), dtype=int32, numpy=6>



On subsequent calls TensorFlow only executes the optimized graph, skipping any non-TensorFlow steps. Below, note that `my_func` doesn't print _tracing_ since `print` is a Python function, not a TensorFlow function.


```python
x = tf.constant([10, 9, 8])
my_func(x)
```




    <tf.Tensor: shape=(), dtype=int32, numpy=27>



A graph may not be reusable for inputs with a different _signature_ (`shape` and `dtype`), so a new graph is generated instead:


```python
x = tf.constant([10.0, 9.1, 8.2], dtype=tf.float32)
my_func(x)
```

    Tracing...
    
    




    <tf.Tensor: shape=(), dtype=float32, numpy=27.3>



These captured graphs provide two benefits:

* In many cases they provide a significant speedup in execution (though not this trivial example).
* You can export these graphs, using `tf.saved_model`, to run on other systems like a [server](https://www.tensorflow.org/tfx/serving/docker) or a [mobile device](https://www.tensorflow.org/lite/guide), no Python installation required.

## Modules, layers, and models

`tf.Module` is a class for managing your `tf.Variable` objects, and the `tf.function` objects that operate on them. The `tf.Module` class is necessary to support two significant features:

1. You can save and restore the values of your variables using `tf.train.Checkpoint`. This is useful during training as it is quick to save and restore a model's state.
2. You can import and export the `tf.Variable` values _and_ the `tf.function` graphs using `tf.saved_model`. This allows you to run your model independently of the Python program that created it.

Here is a complete example exporting a simple `tf.Module` object:


```python
class MyModule(tf.Module):
  def __init__(self, value):
    self.weight = tf.Variable(value)

  @tf.function
  def multiply(self, x):
    return x * self.weight
```


```python
mod = MyModule(3)
mod.multiply(tf.constant([1, 2, 3]))
```




    <tf.Tensor: shape=(3,), dtype=int32, numpy=array([3, 6, 9])>



Save the `Module`:


```python
save_path = './saved'
tf.saved_model.save(mod, save_path)
```

    INFO:tensorflow:Assets written to: ./saved\assets
    

The resulting SavedModel is independent of the code that created it. You can load a SavedModel from Python, other language bindings, or [TensorFlow Serving](https://www.tensorflow.org/tfx/serving/docker). You can also convert it to run with [TensorFlow Lite](https://www.tensorflow.org/lite/guide) or [TensorFlow JS](https://www.tensorflow.org/js/guide).


```python
reloaded = tf.saved_model.load(save_path)
reloaded.multiply(tf.constant([1, 2, 3]))
```




    <tf.Tensor: shape=(3,), dtype=int32, numpy=array([3, 6, 9])>



The `tf.keras.layers.Layer` and `tf.keras.Model` classes build on `tf.Module` providing additional functionality and convenience methods for building, training, and saving models. Some of these are demonstrated in the next section.

## Training loops

Now put this all together to build a basic model and train it from scratch.

First, create some example data. This generates a cloud of points that loosely follows a quadratic curve:


```python
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rcParams['figure.figsize'] = [9, 6]
```


```python
x = tf.linspace(-2, 2, 209)
x = tf.cast(x, tf.float32)

def f(x):
  y = x**2 + 2*x - 5
  return y

y = f(x) + tf.random.normal(shape=[209])

plt.plot(x.numpy(), y.numpy(), '.', label='Data')
plt.plot(x, f(x),  label='Ground truth')
plt.legend();
```


    
![png](output_115_0.png)
    


Create a model:


```python
class Model(tf.keras.Model):
  def __init__(self, units):
    super().__init__()
    self.dense1 = tf.keras.layers.Dense(units=units,
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.random.normal,
                                        bias_initializer=tf.random.normal)
    self.dense2 = tf.keras.layers.Dense(1)

  def call(self, x, training=True):
    # For Keras layers/models, implement `call` instead of `__call__`.
    x = x[:, tf.newaxis]
    x = self.dense1(x)
    x = self.dense2(x)
    return tf.squeeze(x, axis=1)
```


```python
model = Model(64)
```


```python
plt.plot(x.numpy(), y.numpy(), '.', label='data')
plt.plot(x, f(x),  label='Ground truth')
plt.plot(x, model(x), label='Untrained predictions')
plt.title('Before training')
plt.legend();
```


    
![png](output_119_0.png)
    


Write a basic training loop:


```python
variables = model.variables

optimizer = tf.optimizers.SGD(learning_rate=0.001)

for step in range(1000):
  with tf.GradientTape() as tape:
    prediction = model(x)
    error = (y-prediction)**2
    mean_error = tf.reduce_mean(error)
  gradient = tape.gradient(mean_error, variables)
  optimizer.apply_gradients(zip(gradient, variables))

  if step % 100 == 0:
    print(f'Mean squared error: {mean_error.numpy():0.3f}')
```

    Mean squared error: nan
    Mean squared error: nan
    Mean squared error: nan
    Mean squared error: nan
    Mean squared error: nan
    Mean squared error: nan
    Mean squared error: nan
    Mean squared error: nan
    Mean squared error: nan
    Mean squared error: nan
    


```python
plt.plot(x.numpy(),y.numpy(), '.', label="data")
plt.plot(x, f(x),  label='Ground truth')
plt.plot(x, model(x), label='Trained predictions')
plt.title('After training')
plt.legend();
```


    
![png](output_122_0.png)
    


That's working, but remember that implementations of common training utilities are available in the `tf.keras` module. So consider using those before writing your own. To start with, the `Model.compile` and  `Model.fit` methods implement a training loop for you:


```python
new_model = Model(64)
```


```python
new_model.compile(
    loss=tf.keras.losses.MSE,
    optimizer=tf.optimizers.SGD(learning_rate=0.01))

history = new_model.fit(x, y,
                        epochs=100,
                        batch_size=32,
                        verbose=0)

model.save('./my_model')
```

    INFO:tensorflow:Assets written to: ./my_model\assets
    


```python
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylim([0, max(plt.ylim())])
plt.ylabel('Loss [Mean Squared Error]')
plt.title('Keras training progress');
```


    
![png](output_126_0.png)
    


Refer to [Basic training loops](basic_training_loops.ipynb) and the [Keras guide](https://www.tensorflow.org/guide/keras) for more details.


```python

```
