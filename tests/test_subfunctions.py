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
from miscnn import Data_IO, Preprocessor
from miscnn.data_loading.interfaces import Dictionary_interface
from miscnn.data_loading.sample import Sample
from miscnn.processing.subfunctions import *

#-----------------------------------------------------#
#                Unittest: Subfunctions               #
#-----------------------------------------------------#
class SubfunctionsTEST(unittest.TestCase):
    # Create random imaging and segmentation data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Create imgaging and segmentation data
        img2D = np.random.rand(16, 16) * 255
        img2D = img2D.astype(int)
        img3D = np.random.rand(16, 16, 16) * 255
        img3D = img3D.astype(int)
        seg2D = np.random.rand(16, 16) * 3
        seg2D = seg2D.astype(int)
        seg3D = np.random.rand(16, 16, 16) * 3
        seg3D = seg3D.astype(int)
        # Create testing samples
        self.sample2D = Sample("sample2D", img2D, channels=1, classes=3)
        self.sample2Dseg = Sample("sample2Dseg", img2D, channels=1, classes=3)
        self.sample3D = Sample("sample3D", img3D, channels=1, classes=3)
        self.sample3Dseg = Sample("sample3Dseg", img3D, channels=1, classes=3)
        # Add segmentation to seg samples
        self.sample2Dseg.add_segmentation(seg2D)
        self.sample3Dseg.add_segmentation(seg3D)

    #-------------------------------------------------#
    #                Base Functionality               #
    #-------------------------------------------------#
    # Run multiple or none subfunctions during preprocessing
    def test_SUBFUNCTIONS_preprocessing(self):
        ds = dict()
        for i in range(0, 10):
            img = np.random.rand(16, 16, 16) * 255
            img = img.astype(int)
            seg = np.random.rand(16, 16, 16) * 3
            seg = seg.astype(int)
            sample = (img, seg)
            ds["TEST.sample_" + str(i)] = sample
        io_interface = Dictionary_interface(ds, classes=3, three_dim=True)
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="tmp.miscnn.")
        tmp_batches = os.path.join(self.tmp_dir.name, "batches")
        dataio = Data_IO(io_interface, input_path="", output_path="",
                         batch_path=tmp_batches, delete_batchDir=False)
        sf = [Resize((8,8,8)), Normalization(), Clipping(min=-1.0, max=0.0)]
        pp = Preprocessor(dataio, batch_size=1, prepare_subfunctions=False,
                          analysis="fullimage", subfunctions=sf)
        sample_list = dataio.get_indiceslist()
        batches = pp.run(sample_list, training=True, validation=False)
        for i in range(0, 10):
            img = batches[i][0]
            seg = batches[i][1]
            self.assertEqual(img.shape, (1,8,8,8,1))
            self.assertEqual(seg.shape, (1,8,8,8,3))
            self.assertTrue(np.min(img) >= -1.75 and np.max(img) <= 0.75)
        self.tmp_dir.cleanup()

    # Run multiple or none subfunctions during postprocessing
    def test_SUBFUNCTIONS_postprocessing(self):
        ds = dict()
        for i in range(0, 10):
            img = np.random.rand(16, 16, 16) * 255
            img = img.astype(int)
            seg = np.random.rand(16, 16, 16) * 3
            seg = seg.astype(int)
            sample = (img, seg)
            ds["TEST.sample_" + str(i)] = sample
        io_interface = Dictionary_interface(ds, classes=3, three_dim=True)
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="tmp.miscnn.")
        tmp_batches = os.path.join(self.tmp_dir.name, "batches")
        dataio = Data_IO(io_interface, input_path="", output_path="",
                         batch_path=tmp_batches, delete_batchDir=False)
        sf = [Resize((9,9,9)), Normalization(), Clipping(min=-1.0, max=0.0)]
        pp = Preprocessor(dataio, batch_size=1, prepare_subfunctions=False,
                          analysis="patchwise-grid", subfunctions=sf,
                          patch_shape=(4,4,4))
        sample_list = dataio.get_indiceslist()
        for index in sample_list:
            sample = dataio.sample_loader(index)
            for sf in pp.subfunctions:
                sf.preprocessing(sample, training=False)
            pp.cache["shape_" + str(index)] = sample.img_data.shape
            sample.seg_data = np.random.rand(9, 9, 9) * 3
            sample.seg_data = sample.seg_data.astype(int)
            sample.seg_data = to_categorical(sample.seg_data, num_classes=3)
            data_patches = pp.analysis_patchwise_grid(sample, training=True,
                                                      data_aug=False)
            seg_list = []
            for i in range(0, len(data_patches)):
                seg_list.append(data_patches[i][1])
            seg = np.stack(seg_list, axis=0)
            self.assertEqual(seg.shape, (27,4,4,4,3))
            pred = pp.postprocessing(index, seg)
            self.assertEqual(pred.shape, (16,16,16))
        self.tmp_dir.cleanup()

    # Run prepare subfunction of Preprocessor
    def test_SUBFUNCTIONS_prepare(self):
        ds = dict()
        for i in range(0, 10):
            img = np.random.rand(16, 16, 16) * 255
            img = img.astype(int)
            seg = np.random.rand(16, 16, 16) * 3
            seg = seg.astype(int)
            sample = (img, seg)
            ds["TEST.sample_" + str(i)] = sample
        io_interface = Dictionary_interface(ds, classes=3, three_dim=True)
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="tmp.miscnn.")
        tmp_batches = os.path.join(self.tmp_dir.name, "batches")
        dataio = Data_IO(io_interface, input_path="", output_path="",
                         batch_path=tmp_batches, delete_batchDir=False)
        sf = [Resize((8,8,8)), Normalization(), Clipping(min=-1.0, max=0.0)]
        pp = Preprocessor(dataio, batch_size=1, prepare_subfunctions=True,
                          analysis="fullimage", subfunctions=sf)
        sample_list = dataio.get_indiceslist()
        pp.run_subfunctions(sample_list, training=True)
        batches = pp.run(sample_list, training=True, validation=False)
        self.assertEqual(len(os.listdir(tmp_batches)), 10)
        for i in range(0, 10):
            file_prepared_subfunctions = os.path.join(tmp_batches,
                    str(pp.data_io.seed) + ".TEST.sample_" + str(i) + ".pickle")
            self.assertTrue(os.path.exists(file_prepared_subfunctions))
            img = batches[i][0]
            seg = batches[i][1]
            self.assertIsNotNone(img)
            self.assertIsNotNone(seg)
            self.assertEqual(img.shape, (1,8,8,8,1))
            self.assertEqual(seg.shape, (1,8,8,8,3))
        self.tmp_dir.cleanup()

    #-------------------------------------------------#
    #                  Normalization                  #
    #-------------------------------------------------#
    def test_SUBFUNCTIONS_NORMALIZATION_preprocessing(self):
        pass

    def test_SUBFUNCTIONS_NORMALIZATION_postprocessing(self):
        pass
