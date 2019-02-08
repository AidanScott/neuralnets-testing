
#training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
#training_set_outputs = array([[0, 1, 1, 0]]).T
#
#random.seed(1)
#
#synaptic_weights = 2 * random.random((3, 1)) - 1
#
#for iteration in range(10000):
#    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
#
#    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
#
#print( 1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights)))))


#class Network:
#
#    class Node:
#
#        class Edge:
#            def __init__(self, weight, value, source):
#                self.weight = weight
#                self.value = value
#                self.source = source
#
#        bias = 1
#
#        def __init__(self, inputs):
#            self.inputs = inputs
#
#        def output():
#            for var in self.inputs:
#                out += var.value*var.weights
#
#            return 1 / (1 + exp (-(out-bias)))
#
#        def __init__(self):
            
#from numpy import exp, array, random, dot
#
#
#class NeuralNetwork():
#    def __init__(self,synaptic_weights):
#        # Seed the random number generator, so it generates the same numbers
#        # every time the program runs.
#        random.seed(1)
#
#        # We model a single neuron, with 3 input connections and 1 output connection.
#        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
#        # and mean 0.
#        self.synaptic_weights = synaptic_weights
#
#    # The Sigmoid function, which describes an S shaped curve.
#    # We pass the weighted sum of the inputs through this function to
#    # normalise them between 0 and 1.
#    def __sigmoid(self, x):
#        return 1 / (1 + exp(-x))
#
#    # The derivative of the Sigmoid function.
#    # This is the gradient of the Sigmoid curve.
#    # It indicates how confident we are about the existing weight.
#    def __sigmoid_derivative(self, x):
#        return x * (1 - x)
#
#    # We train the neural network through a process of trial and error.
#    # Adjusting the synaptic weights each time.
#    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
#        for iteration in range(number_of_training_iterations):
#            # Pass the training set through our neural network (a single neuron).
#            output = self.think(training_set_inputs)
#
#            # Calculate the error (The difference between the desired output
#            # and the predicted output).
#            error = training_set_outputs - output
#
#            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
#            # This means less confident weights are adjusted more.
#            # This means inputs, which are zero, do not cause changes to the weights.
#            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
#
#            # Adjust the weights.
#            self.synaptic_weights += adjustment
#
#    # The neural network thinks.
#    def think(self, inputs):
#        # Pass inputs through our neural network (our single neuron).
#        return self.__sigmoid(dot(inputs, self.synaptic_weights))
#
#
#if __name__ == "__main__":
#
#    weights = 2 * random.random((3, 1)) - 1
#
#    #Intialise a single neuron neural network.
#    neural_network = NeuralNetwork(weights)
#
#    print ("Random starting synaptic weights: ")
#    print (neural_network.synaptic_weights)
#
#    # The training set. We have 4 examples, each consisting of 3 input values
#    # and 1 output value.
#    training_set_inputs = array([[0, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 1, 0], [0, 0, 1]])
#    training_set_outputs = array([[0, 1, 0, 1, 1, 1, 0]]).T
#
#    # Train the neural network using a training set.
#    # Do it 10,000 times and make small adjustments each time.
#    neural_network.train(training_set_inputs, training_set_outputs, 10000)
#
#    print ("New synaptic weights after training: ")
#    print (neural_network.synaptic_weights)
#
#    # Test the neural network with a new situation.
#    print ("Considering new situation [0, 1, 0] -> ?: ")
#    print(neural_network.think(array([0, 1, 0])) > 0.999)


from numpy import exp, array, random, dot, column_stack

class OneLayerNetwork:

    #constructor, sets weights from arrays created in main
    def __init__(self, weights1, weights2):
        
        self.weights = [weights1, weights2]

    #sigmoid function; normalizes the sum of inputs multiplied by weights to a number between 1 and 0
    def sigmoid(self,x):
        return 1/(1+exp(-x))

    #deriviative of sigmoid, used to impliment psuedo-gradient descent
    def sigmoidDeriviative(self,x):
        return x*(1-x)
    
    #the hard part
    def train(self,training_set_inputs, training_set_outputs, number_of_training_iterations):
       for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
 #           print()

#            print("Output Matrix from the first layer:\n",second_layer_inputs,"\n")

            outputs = self.think(training_set_inputs)

#            print("Output matrix from the second layer:\n",outputs_second_layer,"\n")

            # Calculate the error matrix (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - outputs[0]
#            print("Error matrix:\n",error,"\n")

            # Multiply the error by the input and again by the gradient of the Sigmoid curve. 
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.

            #MY COMMENTS
            #Essentially simplified gradient descent?? only thing is that sigmoid deriviative should be taken at the non-sigmoided output, shouldn't it?
            #I suppose these would be essentially equivalent given minimal change in slope for values greater than 1/less than -1?
            adjustment = dot(outputs[0].T, error * self.sigmoidDeriviative(outputs[0]))

            #Backpropegation takes the deriviative of change in error with respect to the input; 
            #IE the error function of the input, so we should be able to create a new adjustment matrix for the first layer
            change_in_error_with_input = dot(self.weights[1].T,error*self.sigmoidDeriviative(outputs[0]))

            #that is the  change in error multiplied with the sigmoid deriviative of the outputs of the first layer 
            #all dotted with the weights of the first layer... which I'm gonna need to convert into a proper matrix...
            #hmm, or maybe not?

            # Adjust the weights.
#            print("Current weights at the Second Layer:\n",self.weights[1],"\n")
#            print("Adustment Matrix:\n",adjustment,"\n")
            self.weights[1] += adjustment
#            print("Weights after adjustment:\n",self.weights[1],"\n")

            #Backpropegate adjustment.
            #below matrix should be the backpropegation matrix for input weights
            backprop = self.backprop(change_in_error_with_input,training_inputs,outputs[1])

#            print("\nBackpropegation Matrix:",backprop,"\n")

#            print("Current layer 1 weights:\n",self.weights[0][0],"\n",self.weights[0][1],"\n",self.weights[0][2],"\n",self.weights[0][3],"\n")

            self.weights[0][0] = self.weights[0][0] + backprop[0,:].reshape(-1,1)
            self.weights[0][1] = self.weights[0][1] + backprop[1,:].reshape(-1,1)
            self.weights[0][2] = self.weights[0][2] + backprop[2,:].reshape(-1,1)
            self.weights[0][3] = self.weights[0][3] + backprop[3,:].reshape(-1,1)

#            print("Adjusted layer 1 weights:\n",self.weights[0][0],"\n",self.weights[0][1],"\n",self.weights[0][2],"\n",self.weights[0][3],"\n")


    def think(self,inputs):
        #Pass inputs through the network.
        output1 = self.sigmoid(dot(inputs, self.weights[0][0]))
        output2 = self.sigmoid(dot(inputs, self.weights[0][1]))
        output3 = self.sigmoid(dot(inputs, self.weights[0][2]))
        output4 = self.sigmoid(dot(inputs, self.weights[0][3]))

        second_layer_inputs = column_stack((output1,output2,output3,output4))

        return self.sigmoid(dot(second_layer_inputs,self.weights[1])), second_layer_inputs

    def backprop(self, error, input, output):
        return dot(error*self.sigmoidDeriviative(output),input)


if __name__ == "__main__":
    random.seed()

    #set random initial weights to connections between inputs and hidden layer (w1) and hidden layer and outputs (w2)
    w1 = [2 * random.random((2,1)) - 1, 2 * random.random((2,1)) - 1, 2 * random.random((2,1)) - 1, 2 * random.random((2,1)) - 1]
    w2 = 2 * random.random((4,1)) - 1

    #training data
    training_inputs = array([[0,0], [1,0], [0,1], [1,1]])
    training_outputs = array([[0,1,1,1]]).T
    test_input = array([[0,0]])

    #normally only takes two inputs, need to do some testing though
    neuralNet = OneLayerNetwork(w1,w2)

    neuralNet.train(training_inputs,training_outputs,20000)

    print(neuralNet.think(test_input))