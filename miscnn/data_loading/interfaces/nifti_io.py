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
# External libraries
import os
import nibabel as nib
import re
import numpy as np
import warnings
# Internal libraries/scripts
from miscnn.data_loading.interfaces.abstract_io import Abstract_IO

#-----------------------------------------------------#
#                 NIfTI I/O Interface                 #
#-----------------------------------------------------#
""" Data I/O Interface for NIfTI files. The Neuroimaging Informatics Technology Initiative file format
    is designed to contain brain images from e.g. magnetic resonance tomography. Nevertheless, it is
    currently broadly used for any 3D medical image data.

Code source heavily modified from the Kidney Tumor Segmentation Challenge 2019 git repository:
https://github.com/neheller/kits19
"""
class NIFTI_interface(Abstract_IO):
    # Class variable initialization
    def __init__(self, channels=1, classes=2, three_dim=True, pattern=None):
        self.data_directory = None
        self.channels = channels
        self.classes = classes
        self.three_dim = three_dim
        self.pattern = pattern
        self.cache = dict()

    #---------------------------------------------#
    #                  initialize                 #
    #---------------------------------------------#
    # Initialize the interface and return number of samples
    def initialize(self, input_path):
        # Resolve location where imaging data should be living
        if not os.path.exists(input_path):
            raise IOError(
                "Data path, {}, could not be resolved".format(str(input_path))
            )
        # Cache data directory
        self.data_directory = input_path
        # Identify samples
        sample_list = os.listdir(input_path)
        # IF pattern provided: Remove every file which does not match
        if self.pattern != None and isinstance(self.pattern, str):
            for i in reversed(range(0, len(sample_list))):
                if not re.fullmatch(self.pattern, sample_list[i]):
                    del sample_list[i]
        # Return sample list
        return sample_list

    #---------------------------------------------#
    #                  load_image                 #
    #---------------------------------------------#
    # Read a volume NIFTI file from the data directory
    def load_image(self, index):
        # Make sure that the image file exists in the data set directory
        img_path = os.path.join(self.data_directory, index)
        if not os.path.exists(img_path):
            raise ValueError(
                "Image could not be found \"{}\"".format(img_path)
            )
        # Load volume from NIFTI file
        vol = nib.load(os.path.join(img_path, "imaging.nii.gz"))
        # Transform NIFTI object to numpy array
        vol_data = vol.get_fdata()
        # Save spacing in cache
        self.cache[index] = vol.affine
        # Return volume
        return vol_data

    #---------------------------------------------#
    #              load_segmentation              #
    #---------------------------------------------#
    # Read a segmentation NIFTI file from the data directory
    def load_segmentation(self, index):
        # Make sure that the segmentation file exists in the data set directory
        seg_path = os.path.join(self.data_directory, index)
        if not os.path.exists(seg_path):
            raise ValueError(
                "Segmentation could not be found \"{}\"".format(seg_path)
            )
        # Load segmentation from NIFTI file
        seg = nib.load(os.path.join(seg_path, "segmentation.nii.gz"))
        # Transform NIFTI object to numpy array
        seg_data = seg.get_fdata()
        # Return segmentation
        return seg_data

    #---------------------------------------------#
    #               load_prediction               #
    #---------------------------------------------#
    # Read a prediction NIFTI file from the MIScnn output directory
    def load_prediction(self, index, output_path):
        # Resolve location where data should be living
        if not os.path.exists(output_path):
            raise IOError(
                "Data path, {}, could not be resolved".format(str(output_path))
            )
        # Parse the provided index to the prediction file name
        pred_file = str(index) + ".nii.gz"
        pred_path = os.path.join(output_path, pred_file)
        # Make sure that prediction file exists under the prediction directory
        if not os.path.exists(pred_path):
            raise ValueError(
                "Prediction could not be found \"{}\"".format(pred_path)
            )
        # Load prediction from NIFTI file
        pred = nib.load(pred_path)
        # Transform NIFTI object to numpy array
        pred_data = pred.get_fdata()
        # Return prediction
        return pred_data

    #---------------------------------------------#
    #                 load_details                #
    #---------------------------------------------#
    # Parse slice thickness
    def load_details(self, i):
        # Parse voxel spacing from affinity matrix of NIfTI
        spacing_matrix = self.cache[i][:3,:3]
        # Identify correct spacing diagonal
        diagonal_negative = np.diag(spacing_matrix)
        diagonal_positive = np.diag(spacing_matrix[::-1,:])
        if np.count_nonzero(diagonal_negative) != 1:
            spacing = diagonal_negative
        elif np.count_nonzero(diagonal_positive) != 1:
            spacing = diagonal_positive
        else:
            warnings.warn("Affinity matrix of NIfTI volume can not be parsed.")
        # Calculate absolute values for voxel spacing
        spacing = np.absolute(spacing)
        # Delete cached spacing
        del self.cache[i]
        # Return detail dictionary
        return {"spacing":spacing}

    #---------------------------------------------#
    #               save_prediction               #
    #---------------------------------------------#
    # Write a segmentation prediction into in the NIFTI file format
    def save_prediction(self, pred, index, output_path):
        # Resolve location where data should be written
        if not os.path.exists(output_path):
            raise IOError(
                "Data path, {}, could not be resolved".format(output_path)
            )
        # Convert numpy array to NIFTI
        nifti = nib.Nifti1Image(pred, None)
        #nifti.get_data_dtype() == pred.dtype
        # Save segmentation to disk
        pred_file = str(index) + ".nii.gz"
        nib.save(nifti, os.path.join(output_path, pred_file))
