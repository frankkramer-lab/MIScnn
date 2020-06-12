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
import numpy as np
from tensorflow.keras.utils import to_categorical
#Internal libraries
from miscnn import Data_Augmentation

#-----------------------------------------------------#
#             Unittest: Data Augmentation             #
#-----------------------------------------------------#
class DataAugmentationTEST(unittest.TestCase):
    # Create random imaging and segmentation data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Create 2D data
        img2D = np.random.rand(1, 16, 16, 1) * 255
        self.img2D = img2D.astype(int)
        seg2D = np.random.rand(1, 16, 16, 1) * 3
        self.seg2D = seg2D.astype(int)
        self.seg2D = to_categorical(self.seg2D, num_classes=3)
        # Create 3D data
        img3D = np.random.rand(1, 16, 16, 16, 1) * 255
        self.img3D = img3D.astype(int)
        seg3D = np.random.rand(1, 16, 16, 16, 1) * 3
        self.seg3D = seg3D.astype(int)
        self.seg3D = to_categorical(self.seg3D, num_classes=3)

    #-------------------------------------------------#
    #                Base Functionality               #
    #-------------------------------------------------#
    # Class Creation
    def test_DATAAUGMENTATION_BASE_create(self):
        data_Aug = Data_Augmentation()
        self.assertIsInstance(data_Aug, Data_Augmentation)
        Data_Augmentation(cycles=5)
        Data_Augmentation(cycles=1, scaling=True, rotations=True,
                          elastic_deform=False, mirror=False, brightness=True,
                          contrast=True, gamma=True, gaussian_noise=True)
    # Run 2D
    def test_DATAAUGMENTATION_BASE_run2D(self):
        data_aug = Data_Augmentation()
        data_aug.config_p_per_sample = 1
        img_aug, seg_aug = data_aug.run(self.img2D, self.seg2D)
        self.assertEqual(img_aug.shape, self.img2D.shape)
        self.assertFalse(np.array_equal(img_aug, self.img2D))
        self.assertEqual(seg_aug.shape, self.seg2D.shape)
        self.assertFalse(np.array_equal(seg_aug, self.seg2D))
        data_aug = Data_Augmentation(cycles=1, scaling=False, rotations=False,
                          elastic_deform=False, mirror=False, brightness=False,
                          contrast=False, gamma=False, gaussian_noise=False)
        img_aug, seg_aug = data_aug.run(self.img2D, self.seg2D)
        self.assertTrue(np.array_equal(img_aug, self.img2D))

    # Run 3D
    def test_DATAAUGMENTATION_BASE_run3D(self):
        data_aug = Data_Augmentation()
        data_aug.config_p_per_sample = 1
        img_aug, seg_aug = data_aug.run(self.img3D, self.seg3D)
        self.assertEqual(img_aug.shape, self.img3D.shape)
        self.assertFalse(np.array_equal(img_aug, self.img3D))
        self.assertEqual(seg_aug.shape, self.seg3D.shape)
        self.assertFalse(np.array_equal(seg_aug, self.seg3D))
        data_aug = Data_Augmentation(cycles=1, scaling=False, rotations=False,
                          elastic_deform=False, mirror=False, brightness=False,
                          contrast=False, gamma=False, gaussian_noise=False)
        img_aug, seg_aug = data_aug.run(self.img3D, self.seg3D)
        self.assertTrue(np.array_equal(img_aug, self.img3D))

    # Cycles
    def test_DATAAUGMENTATION_BASE_cycles(self):
        with self.assertRaises(Exception):
            data_aug = Data_Augmentation(cycles=0)
            img_aug, seg_aug = data_aug.run(self.img2D, self.seg2D)
        for i in range(1, 50, 5):
            data_aug = Data_Augmentation(cycles=i)
            img_aug, seg_aug = data_aug.run(self.img2D, self.seg2D)
            self.assertEqual(img_aug.shape[0], i)

    #-------------------------------------------------#
    #                    Parameter                    #
    #-------------------------------------------------#
    def test_DATAAUGMENTATION_parameter_cropping(self):
        data_aug = Data_Augmentation(cycles=1, scaling=False, rotations=False,
                          elastic_deform=False, mirror=False, brightness=False,
                          contrast=False, gamma=False, gaussian_noise=False)
        data_aug.cropping = True
        data_aug.cropping_patch_shape = (4,4,4)
        img_aug, seg_aug = data_aug.run(self.img3D, self.seg3D)
        self.assertEqual(img_aug.shape, (1,4,4,4,1))
        self.assertEqual(seg_aug.shape, (1,4,4,4,3))

    def test_DATAAUGMENTATION_parameter_percentage(self):
        data_aug = Data_Augmentation(cycles=100, scaling=True, rotations=False,
                          elastic_deform=False, mirror=False, brightness=False,
                          contrast=False, gamma=False, gaussian_noise=False)
        data_aug.config_p_per_sample = 0.3
        img_aug, seg_aug = data_aug.run(self.img3D, self.seg3D)
        counter_equal = 0
        for i in range(0, 100):
            is_equal = np.array_equal(img_aug[i], self.img3D[0])
            if is_equal : counter_equal += 1
        ratio = counter_equal / 100
        self.assertTrue(ratio >= 0.5 and ratio <= 0.9)

    def test_DATAAUGMENTATION_parameter_scaling(self):
        data_aug = Data_Augmentation(cycles=1, scaling=True, rotations=False,
                          elastic_deform=False, mirror=False, brightness=False,
                          contrast=False, gamma=False, gaussian_noise=False)
        data_aug.config_p_per_sample = 1
        img_aug, seg_aug = data_aug.run(self.img3D, self.seg3D)
        self.assertFalse(np.array_equal(img_aug, self.img3D))
        self.assertFalse(np.array_equal(seg_aug, self.seg3D))

    def test_DATAAUGMENTATION_parameter_rotation(self):
        data_aug = Data_Augmentation(cycles=1, scaling=False, rotations=True,
                          elastic_deform=False, mirror=False, brightness=False,
                          contrast=False, gamma=False, gaussian_noise=False)
        data_aug.config_p_per_sample = 1
        img_aug, seg_aug = data_aug.run(self.img3D, self.seg3D)
        self.assertFalse(np.array_equal(img_aug, self.img3D))
        self.assertFalse(np.array_equal(seg_aug, self.seg3D))

    def test_DATAAUGMENTATION_parameter_edeform(self):
        data_aug = Data_Augmentation(cycles=1, scaling=False, rotations=False,
                          elastic_deform=True, mirror=False, brightness=False,
                          contrast=False, gamma=False, gaussian_noise=False)
        data_aug.config_p_per_sample = 1
        img_aug, seg_aug = data_aug.run(self.img3D, self.seg3D)
        self.assertFalse(np.array_equal(img_aug, self.img3D))
        self.assertFalse(np.array_equal(seg_aug, self.seg3D))

    def test_DATAAUGMENTATION_parameter_mirror(self):
        data_aug = Data_Augmentation(cycles=1, scaling=False, rotations=False,
                          elastic_deform=False, mirror=True, brightness=False,
                          contrast=False, gamma=False, gaussian_noise=False)
        data_aug.config_p_per_sample = 1
        img_aug, seg_aug = data_aug.run(self.img3D, self.seg3D)
        self.assertFalse(np.array_equal(img_aug, self.img3D))
        self.assertFalse(np.array_equal(seg_aug, self.seg3D))

    def test_DATAAUGMENTATION_parameter_brightness(self):
        data_aug = Data_Augmentation(cycles=1, scaling=False, rotations=False,
                          elastic_deform=False, mirror=False, brightness=True,
                          contrast=False, gamma=False, gaussian_noise=False)
        data_aug.config_p_per_sample = 1
        img_aug, seg_aug = data_aug.run(self.img3D, self.seg3D)
        self.assertFalse(np.array_equal(img_aug, self.img3D))
        self.assertTrue(np.array_equal(seg_aug, self.seg3D))

    def test_DATAAUGMENTATION_parameter_contrast(self):
        data_aug = Data_Augmentation(cycles=1, scaling=False, rotations=False,
                          elastic_deform=False, mirror=False, brightness=False,
                          contrast=True, gamma=False, gaussian_noise=False)
        data_aug.config_p_per_sample = 1
        img_aug, seg_aug = data_aug.run(self.img3D, self.seg3D)
        self.assertFalse(np.array_equal(img_aug, self.img3D))
        self.assertTrue(np.array_equal(seg_aug, self.seg3D))

    def test_DATAAUGMENTATION_parameter_gamma(self):
        data_aug = Data_Augmentation(cycles=1, scaling=False, rotations=False,
                          elastic_deform=False, mirror=False, brightness=False,
                          contrast=False, gamma=True, gaussian_noise=False)
        data_aug.config_p_per_sample = 1
        img_aug, seg_aug = data_aug.run(self.img3D, self.seg3D)
        self.assertFalse(np.array_equal(img_aug, self.img3D))
        self.assertTrue(np.array_equal(seg_aug, self.seg3D))

    def test_DATAAUGMENTATION_parameter_gaussiannoise(self):
        data_aug = Data_Augmentation(cycles=1, scaling=False, rotations=False,
                          elastic_deform=False, mirror=False, brightness=False,
                          contrast=False, gamma=False, gaussian_noise=True)
        data_aug.config_p_per_sample = 1
        img_aug, seg_aug = data_aug.run(self.img3D, self.seg3D)
        self.assertFalse(np.array_equal(img_aug, self.img3D))
        self.assertTrue(np.array_equal(seg_aug, self.seg3D))

    def test_DATAAUGMENTATION_parameter_classification(self):
        data_aug = Data_Augmentation()
        data_aug.config_p_per_sample = 1
        data_aug.seg_augmentation = False
        img_aug, seg_aug = data_aug.run(self.img3D, self.seg3D)
        self.assertFalse(np.array_equal(img_aug, self.img3D))
        self.assertTrue(np.array_equal(seg_aug, self.seg3D))
