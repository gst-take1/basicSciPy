from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

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


print("Unique values in images", np.unique(digits_images))
print("Shape of target", digits.target.shape)


print("Unique values in target", np.unique(digits.target))

## data and images are related
reshaped_images = digits.images.reshape(1797, 64)
print(np.all(reshaped_images == digits.data))

# data visualization
# Figure size (width, height) in inches
#fig = plt.figure(figsize=(6, 6))

# Adjust the subplots
#fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# For each of the 64 (no I changes it to 100) images
# for i in range(100):
#     # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position
#     ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
#     # Display an image at the i-th position
#     ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
#     # label the image with the target value
#     ax.text(0, 7, str(digits.target[i]))

# Show the plot
#plt.show()

# my stuff .. I want to visualize variance in the 8 8 8 cells of first 100 images

sumArr = np.zeros((8,8))
print sumArr
for i in range(100):
    sumArr = sumArr + digits.images[i]

#np.set_printoptions(precision=3)
#print sumArr

print np.array_str(sumArr, precision=2, suppress_small=True)
# Show the plot
#plt.show()

# PCA started:
from sklearn import decomposition
# Create a Randomized PCA model that takes two components
randomized_pca = decomposition.PCA(svd_solver='randomized', n_components=2)

# Fit and transform the data to the model
reduced_data_rpca = randomized_pca.fit_transform(digits.data)

print(reduced_data_rpca.shape)
# Create a regular PCA model
pca = decomposition.PCA(n_components=2)

# Fit and transform the data to the model
reduced_data_pca = pca.fit_transform(digits.data)

# Inspect the shape
reduced_data_pca.shape

# Print out the data
print(reduced_data_rpca)
print(reduced_data_pca)

colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
for i in range(len(colors)):
    x = reduced_data_rpca[:, 0][digits.target == i]
    y = reduced_data_rpca[:, 1][digits.target == i]
    plt.scatter(x, y, c=colors[i])
plt.legend(digits.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title("PCA Scatter Plot")
plt.show()






