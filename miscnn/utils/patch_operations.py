#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2020 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
#External libraries
import numpy as np
import math
from batchgenerators.augmentations.utils import pad_nd_image

#-----------------------------------------------------#
#      Pad and crop patch to desired patch shape      #
#-----------------------------------------------------#
def pad_patch(patch, patch_shape, return_slicer=False):
    # Initialize stat length to overwrite batchgenerators default
    kwargs = {"stat_length": None}
    # Transform prediction from channel-last to channel-first structure
    patch = np.moveaxis(patch, -1, 1)
    # Run padding
    padding_results = pad_nd_image(patch, new_shape=patch_shape,
                                   mode="minimum", return_slicer=return_slicer,
                                   kwargs=kwargs)
    # Return padding results
    if return_slicer:
        # Transform data from channel-first back to channel-last structure
        padded_patch = np.moveaxis(padding_results[0], 1, -1)
        return padded_patch, padding_results[1]
    else:
        # Transform data from channel-first back to channel-last structure
        padding_results = np.moveaxis(padding_results, 1, -1)
        return padding_results

def crop_patch(patch, slicer):
    # Transform prediction from channel-last to channel-first structure
    patch = np.moveaxis(patch, -1, 1)
    # Exclude the number of batches and classes from the slice range
    slicer[0] = slice(None)
    slicer[1] = slice(None)
    # Crop patches according to slicer
    patch_cropped = patch[tuple(slicer)]
    # Transform data from channel-first back to channel-last structure
    patch_cropped = np.moveaxis(patch_cropped, 1, -1)
    # Return cropped patch
    return patch_cropped

#-----------------------------------------------------#
#         Slice and Concatenate Function Hubs         #
#-----------------------------------------------------#
# Slice a matrix
def slice_matrix(array, window, overlap, three_dim):
    if three_dim: return slice_3Dmatrix(array, window, overlap)
    else: return slice_2Dmatrix(array, window, overlap)

# Concatenate a matrix
def concat_matrices(patches, image_size, window, overlap, three_dim):
    if three_dim: return concat_3Dmatrices(patches, image_size, window, overlap)
    else: return concat_2Dmatrices(patches, image_size, window, overlap)

#-----------------------------------------------------#
#          Slice and Concatenate 2D Matrices          #
#-----------------------------------------------------#
# Slice a 2D matrix
def slice_2Dmatrix(array, window, overlap):
    # Calculate strides
    strides = array.strides
    shape = ((array.shape[0] - window[0]) // (window[0] - overlap[0])) + 1, \
            ((array.shape[1] - window[1]) // (window[1] - overlap[1])) + 1, \
            window[0], window[1]
    strides = strides[0] * (window[0] - overlap[0]), strides[1] * (window[1] - overlap[1]), \
              strides[0], strides[1]
    # Slice array
    patches = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    return patches

# Concatenate a list of patches together to a numpy matrix
def concat_2Dmatrices(patches, image_size, window, overlap):
    # Calculate steps
    steps_x = int(math.ceil((image_size[0] - overlap[0]) /
                            float(window[0] - overlap[0])))
    steps_y = int(math.ceil((image_size[1] - overlap[1]) /
                            float(window[1] - overlap[1])))
    # Exception Handling: patch overlap smaller than patches
    if steps_x < 1 : steps_x = 1
    if steps_y < 1 : steps_y = 1

    # Iterate over it x,y,z
    matrix_x = None
    matrix_y = None
    matrix_z = None
    pointer = 0
    for x in range(0, steps_x):
        for y in range(0, steps_y):
            # Calculate pointer from 2D steps to 1D list of patches
            pointer = x*steps_y + y
            # Connect current tmp Matrix Z to tmp Matrix Y
            if y == 0:
                matrix_y = patches[pointer]
            else:
                matrix_p = patches[pointer]
                # Handle y-axis overlap
                slice_overlap = calculate_overlap(y, steps_y, overlap,
                                                  image_size, window, 1)
                matrix_y, matrix_p = handle_overlap(matrix_y, matrix_p,
                                                    slice_overlap,
                                                    axis=1)
                matrix_y = np.concatenate((matrix_y, matrix_p), axis=1)
        # Connect current tmp Matrix Y to final Matrix X
        if x == 0:
            matrix_x = matrix_y
        else:
            # Handle x-axis overlap
            slice_overlap = calculate_overlap(x, steps_x, overlap,
                                              image_size, window, 0)
            matrix_x, matrix_y = handle_overlap(matrix_x, matrix_y,
                                                slice_overlap,
                                                axis=0)
            matrix_x = np.concatenate((matrix_x, matrix_y), axis=0)
    # Return final combined matrix
    return(matrix_x)

#-----------------------------------------------------#
#          Slice and Concatenate 3D Matrices          #
#-----------------------------------------------------#
# Slice a 3D matrix
def slice_3Dmatrix(array, window, overlap):
    strides = array.itemsize * np.array([array.shape[1]*array.shape[2], array.shape[2], 1])
    shape = np.array([(array.shape[0] - window[0]) // (window[0] - overlap[0]) + 1,
                      (array.shape[1] - window[1]) // (window[1] - overlap[1]) + 1,
                      (array.shape[2] - window[2]) // (window[2] - overlap[2]) + 1,
                      window[0], window[1], window[2]])
    strides = np.concatenate((strides, strides[-2:] * np.array([window[1], 1])))
    strides[-3:] = array.itemsize

    patches = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

    return patches.reshape([-1, window[0], window[1], window[2]])

# Concatenate a list of patches together to a numpy matrix
def concat_3Dmatrices(patches, image_size, window, overlap):
    # Calculate steps
    steps_x = int(math.ceil((image_size[0] - overlap[0]) /
                            float(window[0] - overlap[0])))
    steps_y = int(math.ceil((image_size[1] - overlap[1]) /
                            float(window[1] - overlap[1])))
    steps_z = int(math.ceil((image_size[2] - overlap[2]) /
                            float(window[2] - overlap[2])))
    # Exception Handling: patch overlap smaller than patches
    if steps_x < 1 : steps_x = 1
    if steps_y < 1 : steps_y = 1
    if steps_z < 1 : steps_z = 1

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
                    matrix_p = patches[pointer]
                    # Handle z-axis overlap
                    slice_overlap = calculate_overlap(z, steps_z, overlap,
                                                      image_size, window, 2)
                    matrix_z, matrix_p = handle_overlap(matrix_z, matrix_p,
                                                        slice_overlap,
                                                        axis=2)
                    matrix_z = np.concatenate((matrix_z, matrix_p),
                                              axis=2)
            # Connect current tmp Matrix Z to tmp Matrix Y
            if y == 0:
                matrix_y = matrix_z
            else:
                # Handle y-axis overlap
                slice_overlap = calculate_overlap(y, steps_y, overlap,
                                                  image_size, window, 1)
                matrix_y, matrix_z = handle_overlap(matrix_y, matrix_z,
                                                    slice_overlap,
                                                    axis=1)
                matrix_y = np.concatenate((matrix_y, matrix_z), axis=1)
        # Connect current tmp Matrix Y to final Matrix X
        if x == 0:
            matrix_x = matrix_y
        else:
            # Handle x-axis overlap
            slice_overlap = calculate_overlap(x, steps_x, overlap,
                                              image_size, window, 0)
            matrix_x, matrix_y = handle_overlap(matrix_x, matrix_y,
                                                slice_overlap,
                                                axis=0)
            matrix_x = np.concatenate((matrix_x, matrix_y), axis=0)
    # Return final combined matrix
    return(matrix_x)

#-----------------------------------------------------#
#          Subroutines for the Concatenation          #
#-----------------------------------------------------#
# Calculate the overlap of the current matrix slice
def calculate_overlap(pointer, steps, overlap, image_size, window, axis):
            # Overlap: IF last axis-layer -> use special overlap size
            if pointer == steps-1 and not (image_size[axis]-overlap[axis]) \
                                            % (window[axis]-overlap[axis]) == 0:
                current_overlap = window[axis] - \
                                  (image_size[axis] - overlap[axis]) % \
                                  (window[axis] - overlap[axis])
            # Overlap: ELSE -> use default overlap size
            else:
                current_overlap = overlap[axis]
            # Return overlap
            return current_overlap

# Handle the overlap of two overlapping matrices
def handle_overlap(matrixA, matrixB, overlap, axis):
    # Access overllaping slice from matrix A
    idxA = [slice(None)] * matrixA.ndim
    matrixA_shape = matrixA.shape
    idxA[axis] = range(matrixA_shape[axis] - overlap, matrixA_shape[axis])
    sliceA = matrixA[tuple(idxA)]
    # Access overllaping slice from matrix B
    idxB = [slice(None)] * matrixB.ndim
    idxB[axis] = range(0, overlap)
    sliceB = matrixB[tuple(idxB)]
    # Calculate Average prediction values between the two matrices
    # and save them in matrix A
    matrixA[tuple(idxA)] = np.mean(np.array([sliceA, sliceB]), axis=0)
    # Remove overlap from matrix B
    matrixB = np.delete(matrixB, [range(0, overlap)], axis=axis)
    # Return processed matrices
    return matrixA, matrixB
