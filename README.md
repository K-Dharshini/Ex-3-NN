<H3>NAME: DHARSHINI K</H3>
<H3>REG NO: 212223230047</H3>
<H3>EX. NO.3</H3>
<H3>DATE:</H3>
<H2 aligh = center> Implementation of MLP for a non-linearly separable data</H2>
<h3>Aim:</h3>
To implement a perceptron for classification using Python
<H3>Theory:</H3>
Exclusive or is a logical operation that outputs true when the inputs differ.For the XOR gate, the TRUTH table will be as follows:<BR>
<BR>

<B>XOR truth table</B>
![Img1](https://user-images.githubusercontent.com/112920679/195774720-35c2ed9d-d484-4485-b608-d809931a28f5.gif)

XOR is a classification problem, as it renders binary distinct outputs. If we plot the INPUTS vs OUTPUTS for the XOR gate, as shown in figure below:<BR>
<BR>

![Img2](https://user-images.githubusercontent.com/112920679/195774898-b0c5886b-3d58-4377-b52f-73148a3fe54d.gif)

<BR>
The graph plots the two inputs corresponding to their output. Visualizing this plot, we can see that it is impossible to separate the different outputs (1 and 0) using a linear equation.To separate the two outputs using linear equation(s), it is required to draw two separate lines as shown in figure below:<BR>
<BR>

![Img 3](https://user-images.githubusercontent.com/112920679/195775012-74683270-561b-4a3a-ac62-cf5ddfcf49ca.gif)

<BR>
For a problem resembling the outputs of XOR, it was impossible for the machine to set up an equation for good outputs. This is what led to the birth of the concept of hidden layers which are extensively used in Artificial Neural Networks. The solution to the XOR problem lies in multidimensional analysis. We plug in numerous inputs in various layers of interpretation and processing, to generate the optimum outputs.
The inner layers for deeper processing of the inputs are known as hidden layers. The hidden layers are not dependent on any other layers. This architecture is known as Multilayer Perceptron (MLP).<BR>
<BR>

![Img 4](https://user-images.githubusercontent.com/112920679/195775183-1f64fe3d-a60e-4998-b4f5-abce9534689d.gif)

<BR>
The number of layers in MLP is not fixed and thus can have any number of hidden layers for processing. In the case of MLP, the weights are defined for each hidden layer, which transfers the signal to the next proceeding layer.Using the MLP approach lets us dive into more than two dimensions, which in turn lets us separate the outputs of XOR using multidimensional equations.Each hidden unit invokes an activation function, to range down their output values to 0 or The MLP approach also lies in the class of feed-forward Artificial Neural Network, and thus can only communicate in one direction. MLP solves the XOR problem efficiently by visualizing the data points in multi-dimensions and thus constructing an n-variable equation to fit in the output values using back propagation algorithm.<BR>

<h3>Algorithm:</H3>

<B>Step 1:</B> Initialize the input patterns for XOR Gate<BR>
<B>Step 2:</B> Initialize the desired output of the XOR Gate<BR>
<B>Step 3:</B> Initialize the weights for the 2 layer MLP with 2 Hidden neuron  and 1 output neuron<BR>
<B>Step 4:</B> Repeat the  iteration  until the losses become constant and  minimum<BR>
    <B>(i)</B>  Compute the output using forward pass output<BR>
    <B>(ii)</B> Compute the error<BR>
    <B>(iii)</B> Compute the change in weight ‘dw’ by using backward progatation algorithm<BR>
    <B>(iv)</B> Modify the weight as per delta rule<BR>
    <B>(v)</B>  Append the losses in a list<BR>
<B>Step 5:</B> Test for the XOR patterns<BR>

<H3>Program:</H3>

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Perceptron:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self._b = 0.0
        self._w = None
        self.misclassified_samples = []

    def fit(self, x: np.array, y: np.array, n_iter=10):
        self._b = 0.0
        self._w = np.zeros(x.shape[1])
        self.misclassified_samples = []
        for _ in range(n_iter):
            errors = 0
            for xi, yi in zip(x, y):
                update = self.learning_rate * (yi - self.predict(xi))
                self._b += update
                self._w += update * xi
                errors += int(update != 0.0)
            self.misclassified_samples.append(errors)

    def f(self, x: np.array) -> float:
        return np.dot(x, self._w) + self._b

    def predict(self, x: np.array):
        return np.where(self.f(x) >= 0, 1, -1)

# Load and prepare the Iris dataset
df = pd.read_csv("iris.csv")
print(df.head())

y = df.iloc[:, 4].values
x = df.iloc[:, 0:3].values

# 3D visualization of the Iris dataset
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title('Iris data set')
ax.set_xlabel('Sepal length in width (cm)')
ax.set_ylabel('Sepal width in width (cm)')
ax.set_zlabel('Petal length in width (cm)')
ax.scatter(x[:50, 0], x[:50, 1], x[:50, 2], color='red', marker='o', s=4, edgecolor='red', label='Iris Setosa')
ax.scatter(x[50:100, 0], x[50:100, 2], color='blue', marker='^', s=4, edgecolor='blue', label='Iris Versicolour')
ax.scatter(x[100:150, 0], x[100:150, 1], x[100:150, 2], color='green', marker='x', s=4, edgecolor='green', label='Iris Virginica')
plt.legend(loc='upper left')
plt.show()

# Prepare data for binary classification (Setosa vs Versicolour)
x = x[0:100, 0:2]
y = y[0:100]

# 2D visualization of the selected classes
plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x', label='Versicolour')
plt.xlabel('Sepal length')
plt.ylabel('Petal length')
plt.legend(loc='upper left')
plt.show()

# Convert labels to binary values (1 for Setosa, -1 for Versicolour)
y = np.where(y == 'Iris-Setosa', 1, -1)

# Normalize features
x[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
x[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Train the perceptron
classifier = Perceptron(learning_rate=0.01)
classifier.fit(x_train, y_train)

# Evaluate the model
print("accuracy", accuracy_score(classifier.predict(x_test), y_test) * 100)

# Plot training errors over epochs
plt.plot(range(1, len(classifier.misclassified_samples) + 1), classifier.misclassified_samples, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Errors')
plt.show()
```

<H3>Output:</H3>

![image](https://github.com/user-attachments/assets/aebcec36-d42c-4b91-a409-e8fd8baa07de)

![image](https://github.com/user-attachments/assets/4ce7eaf1-d2d5-4b5b-8e47-d5b94ddbf2a8)


<H3> Result:</H3>
Thus, XOR classification problem can be solved using MLP in Python 
