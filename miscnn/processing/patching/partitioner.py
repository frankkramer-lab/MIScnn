#==============================================================================#
#  Author:       Philip Meyer                                                  #
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

from miscnn.processing.patching.patch_operations import slice_matrix, concat_matrices, pad_patch, crop_patch
import numpy as np
from miscnn.processing.data_augmentation import Data_Augmentation


class Partitioner():
    """
    
    analysis (string):                      Modus selection of analysis type. Options:
                                            - "fullimage":      Analysis of complete image data
                                            - "patchwise-crop": Analysis of random cropped patches from the image
                                            - "patchwise-grid": Analysis of patches by splitting the image into a grid
    patch_shape (integer tuple):            Size and shape of a patch. The variable has to be defined as a tuple.
                                            For Example: (64,128,128) for 64x128x128 patch cubes.
                                            Be aware that the x-axis represents the number of slices in 3D volumes.
                                            This parameter will be redundant if fullimage or patchwise-crop analysis is selected!!
    """
    def __init__(self, three_dim, analysis, patch_shape=None, patchwise_overlap = (0,0,0), 
                 patchwise_skip_blanks = False, patchwise_skip_class = 0, *argv, **kwargs):
        
        self.three_dim = three_dim
        
        # Exception: Analysis parameter check
        analysis_types = ["patchwise-crop", "patchwise-grid", "fullimage"]
        if not isinstance(analysis, str) or analysis not in analysis_types:
            raise ValueError('Non existent analysis type in preprocessing.')
        # Exception: Patch-shape parameter check
        if (analysis == "patchwise-crop" or analysis == "patchwise-grid") and \
            not isinstance(patch_shape, tuple):
            raise ValueError("Missing or wrong patch shape parameter for " + \
                             "patchwise analysis.")
        self.analysis = analysis
        self.patch_shape = patch_shape
        
        self.patchwise_overlap = patchwise_overlap                      # In patchwise_analysis, an overlap can be defined between adjuncted patches.
        self.patchwise_skip_blanks = patchwise_skip_blanks              # In patchwise_analysis, patches, containing only the background annotation,
                                                                        # can be skipped with this option. This result into only
                                                                        # training on relevant patches and ignore patches without any information.
        self.patchwise_skip_class = patchwise_skip_class                # Class, which will be skipped if patchwise_skip_blanks is True
        self.cache = dict()                                             # Cache additional information and data for patch assembling after patchwise prediction
    
    
    def patch(self, sample, training, data_augmentation):
        
        index = sample.index
        
        if not training : self.cache[index] = sample
        
        ready_data = None
        
        # Run Fullimage analysis
        if self.analysis == "fullimage":
            ready_data = self.full_image(sample, training, data_augmentation)
        # Run patchwise cropping analysis
        elif self.analysis == "patchwise-crop" and training:
            ready_data = self.patch_crop(sample, data_augmentation)
        # Run patchwise grid analysis
        else:
            if not training:
                self.cache["shape_" + str(index)] = sample.img_data.shape
            ready_data = self.patch_grid(sample, training, data_augmentation)
        
        return ready_data
    
    def unpatch(self, sample, prediction):
        if self.analysis == "patchwise-crop" or \
            self.analysis == "patchwise-grid":
            # Check if patch was padded
            slice_key = "slicer_" + str(sample.index)
            if slice_key in self.cache:
                prediction = crop_patch(prediction, self.cache[slice_key])
            # Load cached shape & Concatenate patches into original shape
            seg_shape = self.cache.pop("shape_" + str(sample.index))
            prediction = concat_matrices(patches=prediction,
                                    image_size=seg_shape,
                                    window=self.patch_shape,
                                    overlap=self.patchwise_overlap,
                                    three_dim=self.three_dim)
        # For fullimages remove the batch axis
        else : prediction = np.squeeze(prediction, axis=0)
        
        return prediction
    
    
    #---------------------------------------------#
    #           Patch-wise grid Analysis          #
    #---------------------------------------------#
    def patch_grid(self, sample, training, data_aug):
        # Slice image into patches
        patches_img = slice_matrix(sample.img_data, self.patch_shape,
                                   self.patchwise_overlap,
                                   self.three_dim)
        if training:
            # Slice segmentation into patches
            patches_seg = slice_matrix(sample.seg_data, self.patch_shape,
                                       self.patchwise_overlap,
                                       self.three_dim)
        else : patches_seg = None
        # Skip blank patches (only background)
        if training and self.patchwise_skip_blanks:
            # Iterate over each patch
            for i in reversed(range(0, len(patches_seg))):
                # IF patch DON'T contain any non background class -> remove it
                if not np.any(patches_seg[i][...,self.patchwise_skip_class] != 1):
                    del patches_img[i]
                    del patches_seg[i]
        # Concatenate a list of patches into a single numpy array
        img_data = np.stack(patches_img, axis=0)
        if training : seg_data = np.stack(patches_seg, axis=0)
        # Pad patches if necessary
        if img_data.shape[1:-1] != self.patch_shape and training:
            img_data = pad_patch(img_data, self.patch_shape,return_slicer=False)
            seg_data = pad_patch(seg_data, self.patch_shape,return_slicer=False)
        elif img_data.shape[1:-1] != self.patch_shape and not training:
            img_data, slicer = pad_patch(img_data, self.patch_shape,
                                         return_slicer=True)
            self.cache["slicer_" + str(sample.index)] = slicer
        # Run data augmentation
        if data_aug and training:
            img_data, seg_data = data_aug.run(img_data, seg_data)
        elif data_aug and not training:
            img_data = data_aug.run_infaug(img_data)
        # Create tuple of preprocessed data
        if training:
            ready_data = list(zip(img_data, seg_data))
        else:
            ready_data = list(zip(img_data))
        # Return preprocessed data tuple
        return ready_data

    #---------------------------------------------#
    #           Patch-wise crop Analysis          #
    #---------------------------------------------#
    def patch_crop(self, sample, data_aug):
        # If skipping blank patches is active
        if self.patchwise_skip_blanks:
            # Slice image and segmentation into patches
            patches_img = slice_matrix(sample.img_data, self.patch_shape,
                                       self.patchwise_overlap,
                                       self.three_dim)
            patches_seg = slice_matrix(sample.seg_data, self.patch_shape,
                                       self.patchwise_overlap,
                                       self.three_dim)
            # Skip blank patches (only background)
            for i in reversed(range(0, len(patches_seg))):
                # IF patch DON'T contain any non background class -> remove it
                if not np.any(patches_seg[i][...,self.patchwise_skip_class] != 1):
                    del patches_img[i]
                    del patches_seg[i]
            # Select a random patch
            pointer = np.random.randint(0, len(patches_img))
            img = patches_img[pointer]
            seg = patches_seg[pointer]
            # Expand image dimension to simulate a batch with one image
            img_data = np.expand_dims(img, axis=0)
            seg_data = np.expand_dims(seg, axis=0)
            # Pad patches if necessary
            if img_data.shape[1:-1] != self.patch_shape:
                img_data = pad_patch(img_data, self.patch_shape,
                                     return_slicer=False)
                seg_data = pad_patch(seg_data, self.patch_shape,
                                     return_slicer=False)
            # Run data augmentation
            if data_aug:
                img_data, seg_data = data_aug.run(img_data,
                                                                seg_data)
        # If skipping blank is not active -> random crop
        else:
            # Access image and segmentation data
            img = sample.img_data
            seg = sample.seg_data
            # If no data augmentation should be performed
            # -> create Data Augmentation instance without augmentation methods
            if data_aug is None:
                cropping_data_aug = Data_Augmentation(cycles=1,
                                            scaling=False, rotations=False,
                                            elastic_deform=False, mirror=False,
                                            brightness=False, contrast=False,
                                            gamma=False, gaussian_noise=False)
            else : cropping_data_aug = data_aug
            # Configure the Data Augmentation instance to cropping
            cropping_data_aug.cropping = True
            cropping_data_aug.cropping_patch_shape = self.patch_shape
            # Expand image dimension to simulate a batch with one image
            img_data = np.expand_dims(img, axis=0)
            seg_data = np.expand_dims(seg, axis=0)
            # Run data augmentation and cropping
            img_data, seg_data = cropping_data_aug.run(img_data, seg_data)
        # Create tuple of preprocessed data
        ready_data = list(zip(img_data, seg_data))
        # Return preprocessed data tuple
        
        return ready_data

    #---------------------------------------------#
    #             Full-Image Analysis             #
    #---------------------------------------------#
    def full_image(self, sample, training, data_aug):
        # Access image and segmentation data
        img = sample.img_data
        if training : seg = sample.seg_data
        # Expand image dimension to simulate a batch with one image
        img_data = np.expand_dims(img, axis=0)
        if training : seg_data = np.expand_dims(seg, axis=0)
        # Run data augmentation
        if data_aug and training:
            img_data, seg_data = data_aug.run(img_data, seg_data)
        elif data_aug and not training:
            img_data = data_aug.run_infaug(img_data)
        # Create tuple of preprocessed data
        if training : ready_data = list(zip(img_data, seg_data))
        else : ready_data = list(zip(img_data))
        # Return preprocessed data tuple
        return ready_data
