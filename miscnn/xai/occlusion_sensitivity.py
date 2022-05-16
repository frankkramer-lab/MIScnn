#==============================================================================#
#  Author:       Philip Meyer, Dennis Hartmann                                 #
#  Copyright:    2022 IT-Infrastructure for Translational Medical Research,    #
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
# -----------------------------------------------------#
#                   Library imports                   #
# -----------------------------------------------------#
# External libraries
import math

from tqdm import tqdm
import numpy as np
from miscnn.utils.visualizer import visualize_samples
from miscnn.data_loading.interfaces import Dictionary_interface
from miscnn import Data_IO


'''
The occlusion-sensitivity outputs the map of the relevance of all non-overlapping areas of pixels of the given size
of the input image.
@type  sample_list: List of Strings
@param sample_list: Names of the image-files
@type  model: miscnn-neural_network.mode.Neural_Network
@param model: Model of the neural network of MISCNN
@type  num_cls: Integer
@param num_cls: Class ID
@type  normalize: Boolean
@param normalize: Determines if to normalize map to 0-255 or not
@type  size: Integer
@param size: Size of the zeroed area (square)
@type  in_dir: String
@param in_dir: Path to the images
@type  out_dir: String
@param out_dir: Path where to save the maps
@type  progress: Boolean
@param progress: Determines whether a progress bar should be displayed
@type  return_data: Boolean
@param return_data: Determines whether the map should be returned
@return: Nothing if return_data==False, Occlusionmap if return_data==True
'''
def compute_occlusion_sensitivity_3D(sample_list, model, num_cls=1, normalize=False, size=1, in_dir="", out_dir="vis",
                                  progress=False, return_data=False):
                                  
    """
    Preparation. generate computation agnostic data and configure pipeline.
    
    """
    pp = model.preprocessor
    interface1 = pp.data_io.interface
    dictionary = {0: ([])}
    interface2 = Dictionary_interface(dictionary, channels=interface1.channels, classes=interface1.classes, three_dim=three_dim)
    
    data_io_base = Data_IO(interface1, in_dir, output_path=f"{out_dir}/predictions", delete_batchDir=False)
    data_io_occ_sens = Data_IO(interface2, in_dir, output_path=f"{out_dir}/predictions", delete_batchDir=False)
    
    skip_blanks = pp.patchwise_skip_blanks
    pp.patchwise_skip_blanks = False
    pp.prepare_batches = False
    pp.prepare_subfunctions = False
    
    
    if progress:
        pbar = tqdm(total=len(sample_list))

    if return_data:
        ret = []

    r = math.floor(size / 2)
    if size == 1:
        r = 0

    for index in sample_list:
        # Initialize data path and create the Data I/O instance for the default interface
        pp.data_io = data_io_base
        # Predict the image with the original image
        b = model.predict([index], return_output=True, activation_output=True)[0][num_cls]
        img = pp.data_io.sample_loader(index).img_data
        oc = np.zeros((img.shape[:-1]))
        oc_cntr = np.zeros((img.shape[:-1]))
        
        for i in range(r, img.shape[0], size):
            for j in range(r, img.shape[1], size):
                for k in range(r, img.shape[2], size):
                    temp = np.copy(img)
                    if size > 2:
                        temp[i - r: i + r + 1, j - r: j + r + 1, k - r: k + r + 1] = 0
                    else:
                        temp[i, j, k] = 0
                    # Create a Data I/O interface for loading the changed input data from RAM
                    
                    #assumes the usage of patching
                    patchshape = pp.partitioner.patch_shape
                    patchstep = [x - y for x, y in zip(patchshape, pp.partitioner.patchwise_overlap)]
                    #compute first overlapping patch
                    i_1 = i - (i % patchstep[0])
                    j_1 = j - (j % patchstep[1])
                    k_1 = k - (k % patchstep[2])
                    
                    #compute last overlapping patch
                    i_2 = i + r + 1
                    j_2 = j + r + 1
                    k_2 = k + r + 1
                    i_2 = i_2 - (i_2 % patchstep[0])
                    j_2 = j_2 - (j_2 % patchstep[1])
                    k_2 = k_2 - (k_2 % patchstep[2])
                    
                    temp = temp[i_1 : i_2 + patchshape[0], j_1 : j_2 + patchshape[1], k_1 : k_2 + patchshape[2]]
                
                    dictionary[0][0][0] = temp #not entirely sure about the casing
                    pp.data_io = data_io_occ_sens
                    # Predict from partly zeroed input loaded from RAM
                    a = model.predict(dictionary, return_output=True, activation_output=True)[0][num_cls]
                    # Calculate the difference of the prediction of the original and the partly zeroed one and divide it by
                    
                    print(a.shape)
                    print(b.shape)
                    delta = np.abs(b[i_1 : i_2 + patchshape[0], j_1 : j_2 + patchshape[1], k_1 : k_2 + patchshape[2]] - a)
                    print(delta.shape)
                    if size > 2:
                        oc[i - r: i + r + 1, j - r: j + r + 1, k - r: k + r + 1] += np.sum(delta) / (oc.shape[0] * oc.shape[1] * oc.shape[2] * delta.shape[-1]) #also average over class distribution change
                        oc_cntr[i - r: i + r + 1, j - r: j + r + 1, k - r: k + r + 1] += 1
                    else:
                        oc[i, j, k] += np.sum(b - a) / (oc.shape[0] * oc.shape[1] * delta.shape[-1])
                        oc_cntr[i, j, k] += 1

        oc = np.asarray(oc / oc_cntr)

        if normalize:
            oc = (oc - np.min(oc)) / np.ptp(oc) * 255

        if not return_data:
            # Save and show data
            pp.data_io = data_io_base
            s = pp.data_io.sample_loader(index, False, False)
            s.index = s.index + ".occlusion"
            s.pred_data = oc.astype(np.uint8)
            visualize_samples([s], out_dir=out_dir, mask_pred=True, colorbar=True)
        else:
            ret.append(oc)

        if progress:
            pbar.update(1)
    
    pp.patchwise_skip_blanks = skip_blanks
    
    if return_data:
        return ret
