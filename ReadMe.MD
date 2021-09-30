#Back Propagation Neural Network

An implementation of neural network back propagation algorithm used to train the XOR function.

There are two major classes:

**1. NeuralNet** (*com.neuralnet.framework*)

This is the neural net framework where the number of input and hidden neurons, activation
functions and learning parameters are configurable.  The only assumptions are single hidden 
layer and single output neuron.  It trains the neural net using back propagation algorithm and
writes the resulting error signals to an output file for graph plotting.  The weights can also 
be exported to a file for subsequent loading back to the neural net.


**2. XOR** (*com.xor*)

  This is the XOR function to be trained by the neural network using back propagation.
There are 4 training samples in the data set : {{0, 0) -> 0, {0, 1) -> 1, {1, 0) -> 1, 
{1, 1) -> 0}.  Bipolar representation can also be used where 0 is replaced by -1.

The following 3 configurations will be tested (α = momentum, ρ = learning rate):

- XOR with binary representation, α = 0, ρ = 0.2

- XOR with bipolar representation, α = 0, ρ = 0.2

- XOR with bipolar representation, α = 0.9, ρ = 0.2

Each epoch (containing 4 sets of training data) will be fed into the neural net repeatedly
until the total mean squared error is less than threshold (0.05).  The total error will be
plotted against the number of epochs to verify the error convergence pattern under different
configurations.

The same neural network framework will be used to train the Robocode tanks using Reinforcement
learning in later parts of the assignment.

###Testing
**Test Driven Development (TDD)** approach is used where the software is made as modular as 
possible via Java methods.  The test cases of each method are written in JUnit that drives the
actual coding of the method.  This ensures each method is tested thoroughly in terms of 
expected functionality and code coverage.