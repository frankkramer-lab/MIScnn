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
import unittest
import tempfile
import os
from os.path import join as opj
import numpy as np
import nibabel as nib
from PIL import Image
#Internal libraries
from miscnn.data_loading.interfaces import Image_interface, NIFTI_interface, \
                                           NIFTIslicer_interface, \
                                           Dictionary_interface

#-----------------------------------------------------#
#             Unittest: Data IO Interfaces            #
#-----------------------------------------------------#
class IO_interfacesTEST(unittest.TestCase):
    # Create random imaging and segmentation data
    @classmethod
    def setUpClass(self):
        # Create image and segmentation
        np.random.seed(1234)
        img = np.random.rand(16, 16, 16) * 255
        self.img = img.astype(int)
        seg = np.random.rand(16, 16, 16) * 3
        self.seg = seg.astype(int)
        # Initialize temporary directory
        self.tmp_data = tempfile.TemporaryDirectory(prefix="tmp.miscnn.",
                                                    suffix=".data")
        # Write PNG sample with image and segmentation
        path_sample_image = opj(self.tmp_data.name, "image")
        os.mkdir(path_sample_image)
        img_pillow = Image.fromarray(img[:,:,0].astype(np.uint8))
        img_pillow.save(opj(path_sample_image, "imaging.png"))
        seg_pillow = Image.fromarray(seg[:,:,0].astype(np.uint8))
        seg_pillow.save(opj(path_sample_image, "segmentation.png"))
        # Write NIfTI sample with image and segmentation
        path_sample_nifti = opj(self.tmp_data.name, "nifti")
        os.mkdir(path_sample_nifti)
        nib.save(nib.Nifti1Image(self.img, None), opj(path_sample_nifti,
                                                     "imaging.nii.gz"))
        nib.save(nib.Nifti1Image(self.seg, None), opj(path_sample_nifti,
                                                      "segmentation.nii.gz"))
    # Delete all temporary files
    @classmethod
    def tearDownClass(self):
        self.tmp_data.cleanup()

    #-------------------------------------------------#
    #                 Image Interface                 #
    #-------------------------------------------------#
    # Class Creation
    def test_IOI_IMAGE_creation(self):
        interface = Image_interface()
    # Initialization
    def test_IOI_IMAGE_initialize(self):
        interface = Image_interface(pattern="image")
        sample_list = interface.initialize(self.tmp_data.name)
        self.assertEqual(len(sample_list), 1)
        self.assertEqual(sample_list[0], "image")
    # Loading Images and Segmentations
    def test_IOI_IMAGE_loading(self):
        interface = Image_interface(pattern="image")
        sample_list = interface.initialize(self.tmp_data.name)
        img = interface.load_image(sample_list[0])
        seg = interface.load_segmentation(sample_list[0])
        self.assertTrue(np.array_equal(img, self.img[:,:,0]))
        self.assertTrue(np.array_equal(seg, self.seg[:,:,0]))
    # NIFTI_interface - Loading and Storage of Predictions
    def test_IOI_IMAGE_predictionhandling(self):
        interface = Image_interface(pattern="image")
        sample_list = interface.initialize(self.tmp_data.name)
        interface.save_prediction(self.seg[:,:,0], "pred.image",
                                  self.tmp_data.name)
        pred = interface.load_prediction("pred.image", self.tmp_data.name)
        self.assertTrue(np.array_equal(pred, self.seg[:,:,0]))

    #-------------------------------------------------#
    #                 NIfTI Interface                 #
    #-------------------------------------------------#
    # Class Creation
    def test_IOI_NIFTI_creation(self):
        interface = NIFTI_interface()
    # Initialization
    def test_IOI_NIFTI_initialize(self):
        interface = NIFTI_interface(pattern="nifti")
        sample_list = interface.initialize(self.tmp_data.name)
        self.assertEqual(len(sample_list), 1)
        self.assertEqual(sample_list[0], "nifti")
    # Loading Images and Segmentations
    def test_IOI_NIFTI_loading(self):
        interface = NIFTI_interface(pattern="nifti")
        sample_list = interface.initialize(self.tmp_data.name)
        img = interface.load_image(sample_list[0])
        seg = interface.load_segmentation(sample_list[0])
        details = interface.load_details(sample_list[0])
        self.assertTrue(np.array_equal(img, self.img))
        self.assertTrue(np.array_equal(seg, self.seg))
    # NIFTI_interface - Loading and Storage of Predictions
    def test_IOI_NIFTI_predictionhandling(self):
        interface = NIFTI_interface(pattern="nifti")
        sample_list = interface.initialize(self.tmp_data.name)
        interface.save_prediction(self.seg, "pred.nifti", self.tmp_data.name)
        pred = interface.load_prediction("pred.nifti", self.tmp_data.name)
        self.assertTrue(np.array_equal(pred, self.seg))

    #-------------------------------------------------#
    #              NIfTI slicer Interface             #
    #-------------------------------------------------#
    # Class Creation
    def test_IOI_NIFTIslicer_creation(self):
        interface = NIFTIslicer_interface()
    # Initialization
    def test_IOI_NIFTIslicer_initialize(self):
        interface = NIFTIslicer_interface(pattern="nifti")
        sample_list = interface.initialize(self.tmp_data.name)
        self.assertEqual(len(sample_list), self.img.shape[2])
        self.assertEqual(sample_list[0], "nifti:#:0")
    # Loading Images and Segmentations
    def test_IOI_NIFTIslicer_loading(self):
        interface = NIFTIslicer_interface(pattern="nifti")
        sample_list = interface.initialize(self.tmp_data.name)
        img = interface.load_image(sample_list[-1])
        seg = interface.load_segmentation(sample_list[-1])
        self.assertTrue(np.array_equal(img, self.img[-1]))
        self.assertTrue(np.array_equal(seg, self.seg[-1]))
    # NIFTI_interface - Loading and Storage of Predictions
    def test_IOI_NIFTIslicer_predictionhandling(self):
        interface = NIFTIslicer_interface(pattern="nifti")
        sample_list = interface.initialize(self.tmp_data.name)
        seg = interface.load_segmentation(sample_list[-1])
        interface.save_prediction(seg, "pred.NIIslice", self.tmp_data.name)
        pred = interface.load_prediction("pred.NIIslice", self.tmp_data.name)
        self.assertTrue(np.array_equal(pred, seg))

    #-------------------------------------------------#
    #               Dictionary Interface              #
    #-------------------------------------------------#
    # Class Creation
    def test_IOI_DICTIONARY_creation(self):
        my_dict = dict()
        my_dict["dict_sample"] = (self.img, self.seg)
        interface = Dictionary_interface(my_dict)
    # Initialization
    def test_IOI_DICTIONARY_initialize(self):
        my_dict = dict()
        my_dict["dict_sample"] = (self.img, self.seg)
        interface = Dictionary_interface(my_dict)
        sample_list = interface.initialize("")
        self.assertEqual(len(sample_list), 1)
        self.assertEqual(sample_list[0], "dict_sample")
    # Loading Images and Segmentations
    def test_IOI_DICTIONARY_loading(self):
        my_dict = dict()
        my_dict["dict_sample"] = (self.img, self.seg)
        interface = Dictionary_interface(my_dict)
        sample_list = interface.initialize("")
        img = interface.load_image(sample_list[0])
        seg = interface.load_segmentation(sample_list[0])
        self.assertTrue(np.array_equal(img, self.img))
        self.assertTrue(np.array_equal(seg, self.seg))
    # NIFTI_interface - Loading and Storage of Predictions
    def test_IOI_DICTIONARY_predictionhandling(self):
        my_dict = dict()
        my_dict["dict_sample"] = (self.img, self.seg)
        interface = Dictionary_interface(my_dict)
        sample_list = interface.initialize("")
        interface.save_prediction(self.seg, "dict_sample", "")
        pred = interface.load_prediction("dict_sample", "")
        self.assertTrue(np.array_equal(pred, self.seg))

#-----------------------------------------------------#
#               Unittest: Main Function               #
#-----------------------------------------------------#
if __name__ == '__main__':
    unittest.main()
