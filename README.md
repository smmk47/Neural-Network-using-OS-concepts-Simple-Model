# Neural-Network-using-OS-concepts-Simple-Model-C-
esign an OS that uses separate processes and threads on a multi-core processor for a neural network. Layers are processes, neurons are threads, and inter-process communication through pipes is used. Backpropagation updates weights and biases across layers, leveraging multi-core processing power



First, the required header files are included:

- iostream provides basic input/output services.
- vector is a container class that provides dynamic arrays.
- unistd.h provides access to POSIX operating system APIs, including pipes.
- pthread.h provides POSIX threads (pthreads) for parallel processing.
- fstream provides file input/output services.

Next, the NeuralNetwork class is defined, with the following member functions:

- NeuralNetwork is the main class that represents the neural network.
NeuralNetwork constructor takes two vectors of integers and floats, which represent the number of neurons in each layer and the weights between them, respectively. It initializes the class variables, creates a mutex lock, and sets up pipes and child processes for parallel processing.
forward_propagation function performs forward propagation for a given input vector. It passes the input values through each layer using pipes and returns the final output value.
- backward_propagation function performs backward propagation for a given output error vector. It passes the error values backwards through each layer using pipes, calculates the f(x1) and f(x2) of the error with respect to the input values, and returns a vector of the resulting values.
layers, neurons, weights, input_pipes, and output_pipes are class variables that represent the number of layers, the number of neurons in each layer, the weights between the neurons, and the pipes used for communication, respectively.
- pthread_mutex_t lock is a mutex lock used to protect shared resources.

Next, there are two auxiliary functions:

- compute_neuron is a thread function that performs a forward propagation calculation for a single neuron. It takes in a pointer to an array of floats that includes the neuron's input values and their corresponding weights, and returns a pointer to a float that represents the neuron's output value.

- read_weights reads in weight values from a file and returns them as a 2D vector. It takes in a filename and a vector of integers representing the number of neurons in each layer of the neural network. It reads the weight values from the file and stores them in the appropriate format in the 2D vector.
After that, the code enters the main function, which does the following:

In the main,
- The user is prompted to enter the number of hidden layers and the number of neurons in each hidden layer.
- The neurons vector is initialized with the input and output neurons, and the number of neurons in each hidden layer.
- The weight values are read from a file named "weights.txt" using the read_weights function.
- An instance of the NeuralNetwork class is created with the neurons and weights vectors.
- A sample input vector is created.
- forward_propagation function is called with the input vector to obtain the output value.
- backward_propagation function is called with the output value to obtain the error values with respect to the input values.
