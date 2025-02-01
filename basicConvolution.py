import numpy as np

image = np.array([
    [1,2,3,4,5],
    [6,7,8,9,10],
    [11,12,13,14,15],
    [16,17,18,19,20],
    [21,22,23,24,25]
])

kernel = np.array([
    [1,0,-1],
    [1,0,-1],
    [1,0,-1]
])

# to get dimensions
h,w = image.shape
kh,kw = kernel.shape

# output dimensions
output_h,output_w = h-kh+1,w-kw+1

# initialize zero matrix with above dimensions
output = np.zeros((output_h,output_w))

# convolution
for i in range(output_h):
    for j in range(output_w):
        region = image[i:i+kh,j:j+kw]
        output[i,j] = np.sum(region*kernel)

print("Convolved image: \n", output)