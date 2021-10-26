#==============================================================================#
#  Author:       Dominik Müller, Philip Meyer                                  #
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
from skimage.util import pad as ski_pad
from skimage.util import view_as_windows
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
def slice_matrix(A, window, overlap, three_dim):
    A = np.ascontiguousarray(A)  #A contigous array is required for restriding for restriding
    
    stride, pad_shape, _ = calcPadding(A.shape, window, overlap)
    
    if any([ a[1] > 0 for a in pad_shape]): 
        A = pad(A, window, overlap)
    
    ret = view_as_windows(A, window + A.shape[len(window):], stride + A.shape[len(window):])
    
    #stack the patches in one dimension for batch computation.
    ret = np.reshape(ret, [np.prod(ret.shape[:len(window)]), *window, *A.shape[len(window):]])
    
    return ret

# Concatenate a matrix
# Important note, this method of equially merging the patches currently doesn't handle the case that more than just adjacentpatches overlap i.e. overlap > patch_size / 2
def concat_matrices(patches, image_size, window, overlap, three_dim):
    if three_dim: return unpatch_3D(patches, image_size, overlap)
    else: return unpatch_2D(patches, image_size, overlap)


def calcPadding(orig_size, patch_size, overlap):
    stride = tuple([s - o for s, o in zip(patch_size, overlap)])
    
    pad_shape = []
    padded_shape = []
    for sh, st in zip(orig_size, stride):
        res = (math.ceil(sh/st) * st) - sh
        resL = math.floor(res/2)
        pad_shape.append((resL, res - resL))
        padded_shape.append(sh + res)
    
    return stride, pad_shape, padded_shape

def pad(A, size, overlap):
    _, pad_shape, _ = calcPadding(A.shape, size, overlap)
    return ski_pad(A, tuple(pad_shape) + tuple([(0, 0)] * (len(A.shape) - len(size))), "minimum")

def crop(A, size, patch_size, overlap):
    print(patch_size)
    _, pad_shape, _ = calcPadding(size, patch_size, overlap)
    
    if (len(pad_shape) >= 3):
        return A[pad_shape[0][0]:-pad_shape[0][1], pad_shape[1][0]:-pad_shape[1][1], pad_shape[2][0]:-pad_shape[2][1]]
    elif (len(pad_shape) >= 2):
        return A[pad_shape[0][0]:-pad_shape[0][1], pad_shape[1][0]:-pad_shape[1][1]]

def unpatch_3D(A, orig_size, overlap):
    stride, pad_shape, padded_shape = calcPadding(orig_size, A.shape[1:], overlap)
    
    extra = A.shape[1 + len(overlap):]
    
    #reconstruct the structure that skimage computed for simplicity
    patch_pattern = tuple([math.floor(sh / st) - 1 if o > 0 else math.floor(sh / st) for sh, st, o in zip(padded_shape, stride, overlap)])
    
    A = np.reshape(A, patch_pattern + tuple([1] * (len(orig_size) - len(patch_pattern))) + A.shape[1:])
    
    res = np.zeros(tuple(padded_shape) + extra + (8,))
    res[True] = np.nan #enter invalid value since not all slots are guaranteed to be used and should be dropped for data merging
    
    #reassemble the image.
    #this can be done theoretically over every dimension but it doubles the required bitset (id) every time in order to maintain all patches in one array simultaneously.
    for x in range(patch_pattern[0]):
        for y in range(patch_pattern[1]):
            for z in range(patch_pattern[2]):
                x_pos = x * stride[0]
                y_pos = y * stride[1]
                z_pos = z * stride[2]
                
                id = ((x + y) % 2) + 2 * (z % 4)
                
                res[x_pos:x_pos + A.shape[4], y_pos:y_pos + A.shape[5], z_pos:z_pos + A.shape[6], ..., id] = A[x, y, z]
    
    #unpad the image according to the calculated values
    if any([ a[1] > 0 for a in pad_shape]): 
        unpad = crop(res, orig_size, A.shape[len(overlap) + 1:], overlap)
    else:
        unpad = res
    
    return np.nanmean(unpad, axis = -1)

def unpatch_2D(A, orig_size, overlap):
    stride, pad_shape, padded_shape = calcPadding(orig_size, A.shape[1:], overlap)
    
    extra = A.shape[1 + len(overlap):]
    
    #reconstruct the structure that skimage computed for simplicity
    patch_pattern = tuple([math.floor(sh / st) - 1 if o > 0 else math.floor(sh / st) for sh, st, o in zip(padded_shape, stride, overlap)])

    A = np.reshape(A, patch_pattern + tuple([1] * (len(orig_size) - len(patch_pattern))) + A.shape[1:])
    
    res = np.zeros(tuple(padded_shape) + extra + (4,))
    res[True] = np.nan #enter invalid value since not all slots are guaranteed to be used and should be dropped for data merging
    #reassemble the image.
    #this can be done theoretically over every dimension but it doubles the required bitset (id) every time in order to maintain all patches in one array simultaneously.
    for x in range(patch_pattern[0]):
        for y in range(patch_pattern[1]):
                x_pos = x * stride[0]
                y_pos = y * stride[1]
                
                id = x % 2 + 2 * (y % 2) #this should generate a pattern that all adjacent patches have different values.
                
                res[x_pos:x_pos + A.shape[3], y_pos:y_pos + A.shape[4], ..., id] = A[x, y]
    
    #unpad the image according to the calculated values
    if any([ a[1] > 0 for a in pad_shape]): 
        unpad = crop(res, orig_size, A.shape[len(overlap) + 1:], overlap)
    else:
        unpad = res
    return np.nanmean(unpad, axis = -1)
    