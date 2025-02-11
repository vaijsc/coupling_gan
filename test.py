import numpy as np
import torch
import random

class_labels = torch.eye(3)[random.randint(3, size=[1])]
print(class_labels)

# # Create arrays with different lengths
# array1 = np.array([1, 2, 3])
# array2 = np.array([4, 5, 6, 7, 8])

# # Save the arrays to a file
# np.savez('my_arrays.npz', a=array1, b=array2)

# # Load the arrays from the file
# loaded_data = np.load('my_arrays.npz')

# loaded_array1 = loaded_data['a']
# loaded_array2 = loaded_data['b']

# print("Original arrays:", array1, array2)
# print("Loaded arrays:", loaded_array1, loaded_array2)
# print(loaded_data)

# x, y = loaded_data