# Gradient Descent for Linear Regression
## Spec
This repository is for linear regression with multiple features using gradient descent. The input2.csv contains a series of data points. Each point is a comma-separated ordered triple, representing age, weight, and height (derived from CDC growth charts data).

The gradient descent algorithm is ran using the following learning rates: α ∈ {0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10}. For each value of α, the algorithm runs for exactly 100 iterations. In addition to the nine learning rates above, I came up with my own choice of value for the learning rate.

The gradient descent code was implemented in a file called problem2.py, which will be executed like so:
```$ python problem2.py input2.csv output2.csv```

