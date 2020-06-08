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
#Internal libraries
from miscnn import Data_IO as DataIO
from miscnn.data_loading.interfaces import Dictionary_interface

#-----------------------------------------------------#
#                  Unittest: Data IO                  #
#-----------------------------------------------------#
class Data_IO(unittest.TestCase):
    # Create random imaging and segmentation data
    @classmethod
    def setUpClass(self):
        # Create imgaging and segmentation data set
        np.random.seed(1234)
        self.dataset = dict()
        for i in range(0, 10):
            img = np.random.rand(16, 16, 16) * 255
            self.img = img.astype(int)
            seg = np.random.rand(16, 16, 16) * 3
            self.seg = seg.astype(int)
            if i == 3 : sample = (self.img, self.seg, self.seg)
            elif i == 5 : sample = (self.img, None, self.seg)
            else : sample = (self.img, self.seg)
            self.dataset["TEST.sample_" + str(i)] = sample
        # Initialize Dictionary IO Interface
        self.io_interface = Dictionary_interface(self.dataset)
        # Initialize temporary directory
        self.tmp_batches = tempfile.TemporaryDirectory(prefix="tmp.miscnn.",
                                                       suffix=".batches")

    # Delete all temporary files
    @classmethod
    def tearDownClass(self):
        pass

    #-------------------------------------------------#
    #                Base Functionality               #
    #-------------------------------------------------#
    # Class Creation
    def test_BASE_create(self):
        data_io = DataIO(self.io_interface, input_path="", output_path="",
                         batch_path=self.tmp_batches, delete_batchDir=False)


    # Obtain sample list
    def test_BASE_getSampleList(self):
        data_io = DataIO(self.io_interface, input_path="", output_path="",
                         batch_path=self.tmp_batches, delete_batchDir=False)
        sample_list = data_io.get_indiceslist()
        self.assertEqual(len(sample_list), 10)
        self.assertIn("TEST.sample_0", sample_list)

    # Sample Loader - Imaging
    def test_BASE_SampleLoader_Imaging(self):
        data_io = DataIO(self.io_interface, input_path="", output_path="",
                         batch_path=self.tmp_batches, delete_batchDir=False)
        sample = data_io.sample_loader("TEST.sample_0", backup=False,
                                       load_seg=False, load_pred=False)
        self.assertTrue(np.array_equal(np.reshape(sample.img_data, (16,16,16)),
                                       self.dataset["TEST.sample_0"][0]))
        self.assertEqual(sample.img_data.shape, (16, 16, 16, 1))

    # Sample Loader - Segmentation
    def test_BASE_SampleLoader_Segmentation(self):
        data_io = DataIO(self.io_interface, input_path="", output_path="",
                         batch_path=self.tmp_batches, delete_batchDir=False)
        sample = data_io.sample_loader("TEST.sample_0", backup=False,
                                       load_seg=True, load_pred=False)
        self.assertTrue(np.array_equal(np.reshape(sample.seg_data, (16,16,16)),
                                       self.dataset["TEST.sample_0"][1]))
        self.assertEqual(sample.seg_data.shape, (16, 16, 16, 1))
        self.assertIsNotNone(sample.img_data)
        self.assertIsNone(sample.pred_data)
        with self.assertRaises(Exception):
            sample = data_io.sample_loader("TEST.sample_5", backup=False,
                                           load_seg=True, load_pred=False)

    # Sample Loader - Prediction
    def test_BASE_SampleLoader_Prediction(self):
        data_io = DataIO(self.io_interface, input_path="", output_path="",
                         batch_path=self.tmp_batches, delete_batchDir=False)
        sample = data_io.sample_loader("TEST.sample_5", backup=False,
                                       load_seg=False, load_pred=True)
        self.assertTrue(np.array_equal(np.reshape(sample.pred_data, (16,16,16)),
                                       self.dataset["TEST.sample_5"][2]))
        self.assertEqual(sample.pred_data.shape, (16, 16, 16, 1))
        self.assertIsNotNone(sample.img_data)
        self.assertIsNone(sample.seg_data)
        with self.assertRaises(Exception):
            sample = data_io.sample_loader("TEST.sample_0", backup=False,
                                           load_seg=False, load_pred=True)

    # Sample Loader - Complete
    def test_BASE_SampleLoader_Combined(self):
        data_io = DataIO(self.io_interface, input_path="", output_path="",
                         batch_path=self.tmp_batches, delete_batchDir=False)
        sample = data_io.sample_loader("TEST.sample_3", backup=False,
                                       load_seg=True, load_pred=True)
        self.assertIsNotNone(sample.img_data)
        self.assertIsNotNone(sample.seg_data)
        self.assertIsNotNone(sample.pred_data)
        self.assertEqual(sample.img_data.shape, sample.seg_data.shape)
        self.assertEqual(sample.seg_data.shape, sample.pred_data.shape)

    # #-------------------------------------------------#
    # #                 Batch Management                #
    # #-------------------------------------------------#
    # # Class Creation
    # def test_BASE_create(self):
    #     interface = Image_interface()
    #
    # #-------------------------------------------------#
    # #                Evaluation Backup                #
    # #-------------------------------------------------#
    # # Class Creation
    # def test_EVALUATION_create(self):
    #     interface = Image_interface()

#-----------------------------------------------------#
#               Unittest: Main Function               #
#-----------------------------------------------------#
if __name__ == '__main__':
    unittest.main()
