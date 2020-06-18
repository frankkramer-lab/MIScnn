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
# Internal libraries/scripts
from miscnn.data_loading.interfaces.abstract_io import Abstract_IO

#-----------------------------------------------------#
#                 NIfTI I/O Interface                 #
#-----------------------------------------------------#
""" Data I/O Interface for NIfTI files. The Neuroimaging Informatics Technology Initiative file format
    is designed to contain brain images from e.g. magnetic resonance tomography. Nevertheless, it is
    currently broadly used for any 3D medical image data.

    In contrast to the normal NIfTI IO interface, the NIfTI slicer IO interface splits the 3D volumes
    into separate 2D images (slices).
    This can be useful if it is desired to apply specific 2D architectures.

    Be aware that this interface defines slices on the first axis.
"""
class NIFTIslicer_interface(Abstract_IO):
    # Class variable initialization
    def __init__(self, channels=1, classes=2, pattern=None):
        self.data_directory = None
        self.channels = channels
        self.classes = classes
        self.three_dim = False
        self.pattern = pattern

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
        volume_list = os.listdir(input_path)
        # IF pattern provided: Remove every file which does not match
        if self.pattern != None and isinstance(self.pattern, str):
            for i in reversed(range(0, len(volume_list))):
                if not re.fullmatch(self.pattern, volume_list[i]):
                    del volume_list[i]
        # Open each volume and obtain the number of slices
        sample_list = []
        for index in volume_list:
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
            # Obtain number of slices
            for slice in range(0, vol_data.shape[0]):
                sample_list.append(index + ":#:" + str(slice))
        # Return sample list
        return sample_list

    #---------------------------------------------#
    #                  load_image                 #
    #---------------------------------------------#
    # Read a slice from the data directory
    def load_image(self, index):
        # Identify volume and slice index
        ind_vol = index.split(":#:")[0]
        ind_slice = index.split(":#:")[1]
        # Make sure that the image file exists in the data set directory
        img_path = os.path.join(self.data_directory, ind_vol)
        if not os.path.exists(img_path):
            raise ValueError(
                "Volume could not be found \"{}\"".format(img_path)
            )
        # Load volume from NIFTI file
        vol = nib.load(os.path.join(img_path, "imaging.nii.gz"))
        # Transform NIFTI object to numpy array
        vol_data = vol.get_fdata()
        # Obtain slice from volume
        img_data = vol_data[int(ind_slice)]
        # Return volume
        return img_data

    #---------------------------------------------#
    #              load_segmentation              #
    #---------------------------------------------#
    # Read a segmentation NIFTI file from the data directory
    def load_segmentation(self, index):
        # Identify volume and slice index
        ind_vol = index.split(":#:")[0]
        ind_slice = index.split(":#:")[1]
        # Make sure that the segmentation file exists in the data set directory
        seg_path = os.path.join(self.data_directory, ind_vol)
        if not os.path.exists(seg_path):
            raise ValueError(
                "Segmentation could not be found \"{}\"".format(seg_path)
            )
        # Load segmentation from NIFTI file
        seg = nib.load(os.path.join(seg_path, "segmentation.nii.gz"))
        # Transform NIFTI object to numpy array
        seg_vol_data = seg.get_fdata()
        # Obtain slice from volume
        seg_data = seg_vol_data[int(ind_slice)]
        # Return segmentation
        return seg_data

    #---------------------------------------------#
    #               load_prediction               #
    #---------------------------------------------#
    # Read a prediction file from the MIScnn output directory
    def load_prediction(self, index, output_path):
        # Resolve location where data should be living
        if not os.path.exists(output_path):
            raise IOError(
                "Data path, {}, could not be resolved".format(str(output_path))
            )
        # Parse the provided index to the prediction file name
        pred_path = os.path.join(output_path, str(index) + ".npy")
        # Make sure that prediction file exists under the prediction directory
        if not os.path.exists(pred_path):
            raise ValueError(
                "Prediction could not be found \"{}\"".format(pred_path)
            )

        # Load prediction from file
        pred_data = np.load(pred_path, allow_pickle=True)
        # Return prediction
        return pred_data

    #---------------------------------------------#
    #                 load_details                #
    #---------------------------------------------#
    def load_details(self, i):
        pass

    #---------------------------------------------#
    #               save_prediction               #
    #---------------------------------------------#
    # Write a segmentation prediction into in the NPY file format
    def save_prediction(self, pred, index, output_path):
        # Resolve location where data should be written
        if not os.path.exists(output_path):
            raise IOError(
                "Data path, {}, could not be resolved".format(output_path)
            )
        # Save segmentation to disk as a NumPy pickle
        pred_path = os.path.join(output_path, str(index) + ".npy")
        np.save(pred_path, pred, allow_pickle=True)
