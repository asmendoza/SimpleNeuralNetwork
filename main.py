import numpy as np

class NeuralNetwork():

  def __init__(self):
    # seed random numbers
    np.random.seed(1)

    #initialize the weights randomly
    self.synaptic_weights = 2 * np.random.random((3,1)) - 1

  # sigmoid function to forward 
  def sigmoid(self, x):
    return 1/ (1 + np.exp(-x))

  # sigmoid derivative to adjust weights
  def sigmoid_derivative(self, x):
    return x * (1 - x)

  def think(self, inputs):
    inputs = inputs.astype(float)
    output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
    return output

  def train(self, training_inputs, training_outputs, training_iterations):
      for iteration in range(training_iterations):
        output = self.think(training_inputs)

        error = training_outputs - output
        
        adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

        self.synaptic_weights += adjustments


if __name__ == "__main__":

  nn = NeuralNetwork()
  
  print('Random starting weights: ')
  print(nn.synaptic_weights)

  #input data sets
  training_inputs = np.array([[0,0,1],
                              [1,1,1],
                              [1,0,1],
                              [0,1,1],
                              [0,1,0]])

  #output data sets
  training_outputs = np.array([[0,1,1,0,0]]).T

  nn.train(training_inputs, training_outputs, 100000)
  
  print('Weights after training: ')
  print(nn.synaptic_weights)

  A = str(input("Input 1: "))
  B = str(input("Input 2: "))
  C = str(input("Input 3: "))

  print('New situation: input data = ', A, B, C)
  print('Output data: ')
  print(nn.think(np.array([A, B, C])))
 

  '''
  print ('training inputs')
  print(training_inputs)

  print('training outputs')
  print(training_outputs)

  print('starting weights')
  print(synaptic_weights)


  print('weights after training')
  print(synaptic_weights)

  print('Output after training')
  print(outputs)
  '''