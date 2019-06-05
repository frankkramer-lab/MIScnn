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

# Concatenate a list of patches together to a numpy matrix
def concat_3Dmatrices(patches, image_size, window, overlap):
    # Calculate steps
    steps_x = int(math.ceil(image_size[0] / float(window[0] - overlap)))
    steps_y = int(math.ceil(image_size[1] / float(window[1])))
    steps_z = int(math.ceil(image_size[2] / float(window[2])))

    # Iterate over it x,y,z
    matrix_x = None
    matrix_y = None
    matrix_z = None
    pointer = 0
    for x in range(0, steps_x):
        for y in range(0, steps_y):
            for z in range(0, steps_z):
                # Calculate pointer from 3D steps to 1D list of patches
                pointer = z + y*steps_z + x*steps_y*steps_z
                # Connect current patch to temporary Matrix Z
                if z == 0:
                    matrix_z = patches[pointer]
                else:
                    matrix_z = np.concatenate((matrix_z, patches[pointer]),
                                              axis=2)
            # Connect current tmp Matrix Z to tmp Matrix Y
            if y == 0:
                matrix_y = matrix_z
            else:
                matrix_y = np.concatenate((matrix_y, matrix_z), axis=1)
        # Connect current tmp Matrix Y to final Matrix X
        if x == 0:
            matrix_x = matrix_y
        else:
            # Overlap: IF last x-layer -> handle special overlap size
            if x == steps_x-1:
                last_overlap = window[0] - (image_size[0] % window[0])
                matrix_x, matrix_y = handle_overlap(matrix_x, matrix_y,
                                                    last_overlap)
            # Overlap: Else -> handle default overlap size
            else:
                matrix_x, matrix_y = handle_overlap(matrix_x, matrix_y,
                                                    overlap)
            matrix_x = np.concatenate((matrix_x, matrix_y), axis=0)
    # Return final combined matrix
    return(matrix_x)

# Handle the overlap of two overlapping matrices
def handle_overlap(matrixA, matrixB, overlap, axis=0):
    # Update the overlapping values in Matrix A
    for i in range(0, overlap):
        sliceA = matrixA[len(matrixA) - overlap + i]
        sliceB = matrixB[i]
        # Edge on matrix B -> keep value from matrix A
        if i == 0:
            sliceA = sliceA
        # Edge on matrix A -> keep value from matrix B
        elif i == overlap-1:
            sliceA = sliceB
        # Center value -> Order = Background < Kidney < Tumor
        else:
            for x in range(0, sliceA.shape[0]):
                for y in range(0, sliceA.shape[1]):
                    sliceA[x][y] = np.max((sliceA[x][y], sliceB[x][y]))
        matrixA[len(matrixA) - overlap + i] = sliceA
    # Remove overlap from Matrix B
    matrixB = np.delete(matrixB, [range(0,overlap)], axis=axis)
    # Return processed matrices
    return matrixA, matrixB
