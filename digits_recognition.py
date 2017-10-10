from sklearn import datasets

digits = datasets.load_digits()

#print(digits);

# LEts see the keys
print(digits.keys())

# for key1 in digits.keys():
#     print(key1)
#     print(digits.__getattr__(key=key1))

data = digits['data']
print("Shape of data", data.shape)

print(data)
# Isolate the `images`
digits_images = digits.images

# Inspect the shape
print("Shape of images", digits_images.shape)
print(digits_images)

# print the value at a random index in the images array
print(digits.images[550]);

import numpy as np
print("Unique values in images", np.unique(digits_images))
print("Shape of target", digits.target.shape)


print("Unique values in target", np.unique(digits.target))

## data and images are related
reshaped_images = digits.images.reshape(1797, 64)
print(np.all(reshaped_images == digits.data))

# data visualization

import matplotlib.pyplot as plt
# Figure size (width, height) in inches
fig = plt.figure(figsize=(6, 6))

# Adjust the subplots
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# For each of the 64 (no I changes it to 100) images
for i in range(100):
    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position
    ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
    # Display an image at the i-th position
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))

# Show the plot
plt.show()





