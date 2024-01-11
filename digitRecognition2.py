from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2




images, labels = get_mnist()
weights_input_hidden = np.random.uniform(-0.5, 0.5, (20, 784))
weights_hidden_output = np.random.uniform(-0.5, 0.5, (10, 20))
bias_input_hidden = np.zeros((20, 1))
bias_hidden_out = np.zeros((10, 1))

learn_rate = 0.01
nr_correct = 0
epochs = 3
#loops through a set number of epochs, how many times we loop through each image
for epoch in range(epochs):
    #loops through all of the images and their labels
    for img, l in zip(images, labels):
        img.shape += (1,)
        l.shape += (1,)
        # Forward propagation input -> hidden
        hidden_layer_pre = bias_input_hidden + weights_input_hidden @ img
        hidden_layer = 1 / (1 + np.exp(-hidden_layer_pre))
        # Forward propagation hidden -> output
        output_pre = bias_hidden_out + weights_hidden_output @ hidden_layer
        output = 1 / (1 + np.exp(-output_pre))

        # Cost / Error calculation
        error = 1 / len(output) * np.sum((output - l) ** 2, axis=0)
        nr_correct += int(np.argmax(output) == np.argmax(l))

        # Backpropagation output -> hidden (cost function derivative)
        delta_output = output - l
        weights_hidden_output += -learn_rate * delta_output @ np.transpose(hidden_layer) #updates the weights between the hidden and output layer
        bias_hidden_out += -learn_rate * delta_output #updates bias weight
        # Backpropagation hidden -> input (activation function derivative)
        delta_h = np.transpose(weights_hidden_output) @ delta_output * (hidden_layer * (1 - hidden_layer)) #shows how much each nueron in the hidden layer contributed to the error
        weights_input_hidden += -learn_rate * delta_h @ np.transpose(img) #upates weights between hidden and input layer
        bias_input_hidden += -learn_rate * delta_h

    # Show accuracy for this epoch
    print(f"Epoch: {epoch+1}, Acc: {round((nr_correct / images.shape[0]) * 100, 2)}%")
    nr_correct = 0

# Show results

while True:
    index = int(input("Enter a number (0 - 59999): "))
    img = images[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img.shape += (1,)
    # Forward propagation input -> hidden
    hidden_layer_pre = bias_input_hidden + weights_input_hidden @ img.reshape(784, 1)
    hidden_layer = 1 / (1 + np.exp(-hidden_layer_pre))
    # Forward propagation hidden -> output
    output_pre = bias_hidden_out + weights_hidden_output @ hidden_layer   
    output = 1 / (1 + np.exp(-output_pre))

    plt.title(f"This digit is most likely a {output.argmax()}")
    plt.show()
    
