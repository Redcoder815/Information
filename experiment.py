# The below code implements the softmax function
# using python and numpy. It takes:
# Input: It takes input array/list of values
# Output: Outputs a array/list of softmax values.


# Importing the required libraries
import numpy as np

# Defining the softmax function
def softmax(values):

    # Computing element wise exponential value
    exp_values = np.exp(values)

    # Computing sum of these values
    exp_values_sum = np.sum(exp_values)

    # Returing the softmax output.
    output_value = exp_values/exp_values_sum
    return output_value


if __name__ == '__main__':

    # Input to be fed
    values = [2, 4, 5, 3]

    # Output achieved
    output = softmax(values)
    print("Softmax Output: ", output)
    print("Sum of Softmax Values: ", np.sum(output))
