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
from miscnn import Data_IO
from miscnn.data_loading.interfaces import Dictionary_interface

#-----------------------------------------------------#
#                  Unittest: Data IO                  #
#-----------------------------------------------------#
class Data_IOTEST(unittest.TestCase):
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
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="tmp.miscnn.")
        self.tmp_batches = os.path.join(self.tmp_dir.name, "batches")

    # Delete all temporary files
    @classmethod
    def tearDownClass(self):
        self.tmp_dir.cleanup()

    #-------------------------------------------------#
    #                Base Functionality               #
    #-------------------------------------------------#
    # Class Creation
    def test_DATAIO_BASE_create(self):
        data_io = Data_IO(self.io_interface, input_path="", output_path="",
                         batch_path=self.tmp_batches, delete_batchDir=False)


    # Obtain sample list
    def test_DATAIO_BASE_getSampleList(self):
        data_io = Data_IO(self.io_interface, input_path="", output_path="",
                         batch_path=self.tmp_batches, delete_batchDir=False)
        sample_list = data_io.get_indiceslist()
        self.assertEqual(len(sample_list), 10)
        self.assertIn("TEST.sample_0", sample_list)

    # Prediction storage
    def test_DATAIO_BASE_savePrediction(self):
        data_io = Data_IO(self.io_interface, input_path="",
                         output_path=os.path.join(self.tmp_dir.name, "pred"),
                         batch_path=self.tmp_batches, delete_batchDir=False)
        sample = data_io.sample_loader("TEST.sample_0", backup=False,
                                       load_seg=True, load_pred=False)
        self.assertIsNone(sample.pred_data)
        data_io.save_prediction(sample.seg_data, sample.index)
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir.name, "pred")))
        sample = data_io.sample_loader("TEST.sample_0", backup=False,
                                       load_seg=True, load_pred=True)
        self.assertTrue(np.array_equal(sample.seg_data, sample.pred_data))

    #-------------------------------------------------#
    #                  Sample Loader                  #
    #-------------------------------------------------#
    # Sample Loader - Imaging
    def test_DATAIO_SampleLoader_Imaging(self):
        data_io = Data_IO(self.io_interface, input_path="", output_path="",
                         batch_path=self.tmp_batches, delete_batchDir=False)
        sample = data_io.sample_loader("TEST.sample_0", backup=False,
                                       load_seg=False, load_pred=False)
        self.assertTrue(np.array_equal(np.reshape(sample.img_data, (16,16,16)),
                                       self.dataset["TEST.sample_0"][0]))
        self.assertEqual(sample.img_data.shape, (16, 16, 16, 1))

    # Sample Loader - Segmentation
    def test_DATAIO_SampleLoader_Segmentation(self):
        data_io = Data_IO(self.io_interface, input_path="", output_path="",
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
    def test_DATAIO_SampleLoader_Prediction(self):
        data_io = Data_IO(self.io_interface, input_path="", output_path="",
                         batch_path=self.tmp_batches, delete_batchDir=False)
        sample = data_io.sample_loader("TEST.sample_5", backup=False,
                                       load_seg=False, load_pred=True)
        self.assertTrue(np.array_equal(np.reshape(sample.pred_data, (16,16,16)),
                                       self.dataset["TEST.sample_5"][2]))
        self.assertEqual(sample.pred_data.shape, (16, 16, 16, 1))
        self.assertIsNotNone(sample.img_data)
        self.assertIsNone(sample.seg_data)
        with self.assertRaises(Exception):
            sample = data_io.sample_loader("TEST.sample_2", backup=False,
                                           load_seg=False, load_pred=True)

    # Sample Loader - Complete
    def test_DATAIO_SampleLoader_Combined(self):
        data_io = Data_IO(self.io_interface, input_path="", output_path="",
                         batch_path=self.tmp_batches, delete_batchDir=False)
        sample = data_io.sample_loader("TEST.sample_3", backup=False,
                                       load_seg=True, load_pred=True)
        self.assertIsNotNone(sample.img_data)
        self.assertIsNotNone(sample.seg_data)
        self.assertIsNotNone(sample.pred_data)
        self.assertEqual(sample.img_data.shape, sample.seg_data.shape)
        self.assertEqual(sample.seg_data.shape, sample.pred_data.shape)

    #-------------------------------------------------#
    #                 Batch Management                #
    #-------------------------------------------------#
    # Batch Storage
    def test_DATAIO_BATCHES_backup(self):
        data_io = Data_IO(self.io_interface, input_path="", output_path="",
                         batch_path=self.tmp_batches, delete_batchDir=False)
        sample = data_io.sample_loader("TEST.sample_0", backup=False,
                                       load_seg=True, load_pred=False)
        data_io.backup_batches(sample.img_data, sample.seg_data, "abc")
        self.assertEqual(len(os.listdir(self.tmp_batches)), 2)
        data_io.batch_cleanup()

    # Batch Loading
    def test_DATAIO_BATCHES_loading(self):
        data_io = Data_IO(self.io_interface, input_path="", output_path="",
                         batch_path=self.tmp_batches, delete_batchDir=False)
        sample = data_io.sample_loader("TEST.sample_0", backup=False,
                                       load_seg=True, load_pred=False)
        data_io.backup_batches(sample.img_data, sample.seg_data, "abc")
        img = data_io.batch_load(pointer="abc", img=True)
        self.assertTrue(np.array_equal(sample.img_data, img))
        seg = data_io.batch_load(pointer="abc", img=False)
        self.assertTrue(np.array_equal(sample.seg_data, seg))
        data_io.batch_cleanup()

    # Batch Cleanup
    def test_DATAIO_BATCHES_cleanup(self):
        data_io = Data_IO(self.io_interface, input_path="", output_path="",
                         batch_path=self.tmp_batches, delete_batchDir=False)
        sample = data_io.sample_loader("TEST.sample_0", backup=False,
                                       load_seg=True, load_pred=False)
        data_io.backup_batches(sample.img_data, sample.seg_data, "abc")
        data_io.backup_batches(sample.img_data, sample.seg_data, "def")
        data_io.backup_batches(sample.img_data, None, pointer="ghi")
        self.assertEqual(len(os.listdir(self.tmp_batches)), 5)
        data_io.batch_cleanup(pointer="def")
        self.assertEqual(len(os.listdir(self.tmp_batches)), 3)
        data_io.batch_cleanup()
        self.assertEqual(len(os.listdir(self.tmp_batches)), 0)

    # Sample Storage
    def test_DATAIO_BATCHES_sampleStorage(self):
        data_io = Data_IO(self.io_interface, input_path="", output_path="",
                         batch_path=self.tmp_batches, delete_batchDir=False)
        sample = data_io.sample_loader("TEST.sample_0", backup=False,
                                       load_seg=True, load_pred=False)
        data_io.backup_sample(sample)
        self.assertEqual(len(os.listdir(self.tmp_batches)), 1)
        data_io.batch_cleanup()

    # Sample Loading
    def test_DATAIO_BATCHES_sampleLoading(self):
        data_io = Data_IO(self.io_interface, input_path="", output_path="",
                         batch_path=self.tmp_batches, delete_batchDir=False)
        sample = data_io.sample_loader("TEST.sample_0", backup=False,
                                       load_seg=True, load_pred=False)
        data_io.backup_sample(sample)
        sample_new = data_io.load_sample_pickle(sample.index)
        data_io.batch_cleanup()
        self.assertTrue(np.array_equal(sample_new.img_data, sample.img_data))
        self.assertTrue(np.array_equal(sample_new.seg_data, sample.seg_data))

#-----------------------------------------------------#
#               Unittest: Main Function               #
#-----------------------------------------------------#
if __name__ == '__main__':
    unittest.main()
