#include <iostream>
#include <vector>
#include <unistd.h>
#include <pthread.h>
#include <fstream>
using namespace std;
class NeuralNetwork {
public:
    NeuralNetwork(vector<int> neurons, vector<std::vector<float>> weights);
    float forward_propagation(std::vector<float> input);
    vector<float> backward_propagation(vector<float> output_error);

    int layers;
    vector<int> neurons;
    vector<std::vector<float>> weights;
    vector<int*> input_pipes;
    vector<int*> output_pipes;
    pthread_mutex_t lock;
};


/*This thread function performs a forward propagation calculation for a single neuron.
It takes in a pointer to an array of floats that includes the neuron's input values and their
corresponding weights, and returns a pointer to a float that represents the neuron's output value.*/
void* compute_neuron(void* arg) {
    float* input = (float*)arg;
    int num_inputs = (int)input[0];
    float weight = *(input + num_inputs + 1);
    float output = 0;
    for (int i = 1; i <= num_inputs; i++) {
        output += input[i] * weight;
    }
    float* output_ptr = new float(output);
    delete[] input;
    return (void*)output_ptr;
}

/*This is the constructor for the NeuralNetwork class. It takes in two vectors of integers and floats,
which represent the number of neurons in each layer and the weights between them, respectively.
It initializes the class variables, creates a mutex lock, and sets up pipes and child processes for parallel processing.*/
NeuralNetwork::NeuralNetwork(std::vector<int> neurons, std::vector<std::vector<float>> weights) {
    this->layers = neurons.size() - 1;
    this->neurons = neurons;
    this->weights = weights;
    pthread_mutex_init(&lock, NULL);
    input_pipes.clear();
    output_pipes.clear();
    for (int i = 0; i < layers; i++) {
        int* input_pipe = new int[2];
        int* output_pipe = new int[2];
        if (pipe(input_pipe) < 0 || pipe(output_pipe) < 0) {
            perror("pipe error");
            exit(1);
        }
        input_pipes.push_back(input_pipe);
        output_pipes.push_back(output_pipe);
        pid_t pid = fork();
        if (pid < 0) {
            perror("fork error");
            exit(1);
        }
        else if (pid == 0) {
            // child process
            close(input_pipe[1]);
            close(output_pipe[0]);
            while (true) {
                float input[neurons[i] + 2];
                read(input_pipe[0], &input, sizeof(float) * (neurons[i] + 2));
                if (input[0] == -1) {
                    // terminate signal received
                    break;
                }
                pthread_t threads[neurons[i + 1]];
                float* outputs[neurons[i + 1]];
                for (int j = 0; j < neurons[i + 1]; j++) {
                    float* data = new float[neurons[i] + 2];
                    data[0] = (float)neurons[i];
                    for (int k = 0; k < neurons[i]; k++) {
                        data[k + 1] = input[k + 1];
                    }
                    data[neurons[i] + 1] = weights[i][j];
                    pthread_create(&threads[j], NULL, &compute_neuron, (void*)data);
                }
                for (int j = 0; j < neurons[i + 1]; j++) {
                    void* output;
                    pthread_join(threads[j], &output);
                    outputs[j] = (float*)output;
                    pthread_mutex_lock(&lock);
                    write(output_pipe[1], outputs[j], sizeof(float));
                    pthread_mutex_unlock(&lock);
                    delete outputs[j];
                }
            }
            close(input_pipe[0]);
            close(output_pipe[1]);
            exit(0);
        }
        else {
            // parent process
            close(input_pipe[0]);
            close(output_pipe[1]);
        }
    }
}

/*This function performs forward propagation for a given input vector.
It passes the input values through each layer using pipes and returns the final output value.*/
float NeuralNetwork::forward_propagation(vector<float> input) {
    float output;
    std::vector<float> curr_input = input;
    for (int i = 0; i < layers; i++) {
        write(input_pipes[i][1], curr_input.data(), sizeof(float) * (neurons[i] + 1));
        vector<float> curr_output(neurons[i + 1]);
        for (int j = 0; j < neurons[i + 1]; j++) {
            // use mutex to protect the shared resource
            pthread_mutex_lock(&lock);
            read(output_pipes[i][0], &curr_output[j], sizeof(float));
            pthread_mutex_unlock(&lock);
        }
        curr_input = curr_output;
        cout << "Forward Propagation: Passing from layer " << i << ": ";
        for (int j = 0; j < neurons[i + 1]; j++) {
            if (j == neurons[i + 1] - 1) {
                output = curr_output[j];
            }
            cout << curr_output[j] << " ";
        }
        cout <<endl;
    }
    // send terminate signal to all child processes except the last one
    int terminate_signal = -1;
    for (int i = 0; i < layers - 1; i++) {
        write(input_pipes[i + 1][1], &terminate_signal, sizeof(int));
    }
    cout << "Forward Propagation output: " <<endl;
    cout << "x = " << output <<endl;
    return output;
}

/*This function performs backward propagation for a given output error vector.
It passes the error values backwards through each layer using pipes,
calculates the f(x1) and f(x2) of the error with respect to the input values, and returns a vector of the resulting values.*/
std::vector<float> NeuralNetwork::backward_propagation(std::vector<float> output_error) {
    std::vector<float> curr_error = output_error;
    for (int i = layers - 1; i >= 0; i--) {
        for (int j = 0; j < neurons[i + 1]; j++) {
            // use mutex to protect the shared resource
            pthread_mutex_lock(&lock);
            write(output_pipes[i][1], &curr_error[j], sizeof(float));
            pthread_mutex_unlock(&lock);
        }
        vector<float> prev_error(neurons[i]);
        for (int j = 0; j < neurons[i]; j++) {
            // use mutex to protect the shared resource
            pthread_mutex_lock(&lock);
            read(input_pipes[i][0], &prev_error[j], sizeof(float));
            pthread_mutex_unlock(&lock);
        }
        curr_error = prev_error;
    }
    cout << "Backward Propagation: " << std::endl;
    for (int i = layers - 1; i >= 0; i--) {
        cout << "Received from layer " << i + 1 << ": " << output_error[0] <<endl;
        cout << "f(x1) = " << (output_error[0] * output_error[0] + output_error[0] + 1) / 2 <<endl;
        cout << "f(x2) = " << (output_error[0] * output_error[0] - output_error[0]) / 2 <<endl;
    }

    vector<float> fx;
    fx.push_back((output_error[0] * output_error[0] + output_error[0] + 1) / 2);
    fx.push_back((output_error[0] * output_error[0] - output_error[0]) / 2);

    return fx;
}

/*This function reads in weight values from a file and returns them as a 2D vector.
It takes in a filename and a vector of integers representing the number of neurons in
each layer of the neural network. It reads the weight values from the file and stores
them in the appropriate format in the 2D vector.*/
std::vector<std::vector<float>> read_weights(const std::string& filename, const std::vector<int>& neurons) {
    std::ifstream infile(filename);
    if (!infile) {
        throw runtime_error("Error opening file: " + filename);
    }

    vector<vector<float>> weights;
    vector<float> row;
    float val;
    int count = 0;
    while (infile >> val)
    {
        row.push_back(val);
        if (row.size() == neurons[count] * neurons[count + 1])
        {
            weights.push_back(row);
            row.clear();
            count++;
        }
    }

    if (weights.size() != neurons.size() - 1) {
        throw runtime_error("Error: incorrect number of weight matrices in file.");
    }

    return weights;
}

int main() {
    int hnum = 0;
    cout << "------------------------ PARALLELIZED NEURAL NETWORK ------------------------" <<endl;
    cout << "Enter the number of hidden layers: ";
    cin >> hnum;
    vector<int> neurons;
    neurons.push_back(2); // 2 input neurons
    cout << "Enter the number of neurons in each hidden layer: " <<endl;
    int x = 0;
    for (int i = 0; i < hnum; i++)
    {
        cout << "Number of neurons in hidden layer " << i << ": ";
        cin >> x;
        neurons.push_back(x);
    }
    neurons.push_back(1); // 1 output neuron

    vector<vector<float>> weights = read_weights("weights.txt", neurons);
    NeuralNetwork nn(neurons, weights);
    vector<float> input = { 0.1, 0.2 };
    float output = nn.forward_propagation(input);
    vector<float> output_error = { output };
    nn.backward_propagation(output_error);
    return 0;
}