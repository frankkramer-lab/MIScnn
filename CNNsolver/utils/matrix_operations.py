import numpy as np
import math

# Slice a 3D matrix
def slice_3Dmatrix(array, window, overlap):
    # Calculate steps
    steps_x = int(math.ceil(len(array) / float(window[0] - overlap)))
    steps_y = int(math.ceil(len(array[0]) / float(window[1])))
    steps_z = int(math.ceil(len(array[0][0]) / float(window[2])))

    # Iterate over it x,y,z
    patches = []
    for x in range(0, steps_x):
        for y in range(0, steps_y):
            for z in range(0, steps_z):
                # Define window edges
                x_start = x*window[0] - x*overlap
                x_end = x_start + window[0]
                y_start = y*window[1]
                y_end = y_start + window[1]
                z_start = z*window[2]
                z_end = z_start + window[2]
                # Adjust ends
                if(x_end > len(array)):
                    # Create an overlapping patch for the last images / edges
                    # to ensure the fixed patch/window sizes
                    x_start = len(array) - window[0]
                    x_end = len(array)
                if(y_end > len(array[0])):
                    y_start = len(array[0])
                    y_end = len(array[0])
                if(z_end > len(array[0][0])):
                    z_start = len(array[0][0])
                    z_end = len(array[0][0])
                # Cut window
                window_cut = array[x_start:x_end,y_start:y_end,z_start:z_end]
                # Reshape window
                window_slice = np.reshape(window_cut, (1,) + window_cut.shape)
                # Add to result list
                patches.append(window_slice)
    return patches

# Concatenate two or more matrices in a list together to a numpy array/matrix
def concat_3Dmatrices(matrix_list, axis=0):
    result_matrix = np.concatenate(matrix_list, axis=axis)
    return result_matrix
