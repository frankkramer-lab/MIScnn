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
from miscnn import Data_IO, Preprocessor, Neural_Network
from miscnn.data_loading.interfaces import Dictionary_interface

#-----------------------------------------------------#
#              Unittest: Neural Network               #
#-----------------------------------------------------#
class NeuralNetworkTEST(unittest.TestCase):
    # Create random imaging and segmentation data
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Create 2D imgaging and segmentation data set
        self.dataset2D = dict()
        for i in range(0, 6):
            img = np.random.rand(16, 16) * 255
            self.img = img.astype(int)
            seg = np.random.rand(16, 16) * 3
            self.seg = seg.astype(int)
            self.dataset2D["TEST.sample_" + str(i)] = (self.img, self.seg)
        # Initialize Dictionary IO Interface
        io_interface2D = Dictionary_interface(self.dataset2D, classes=3,
                                              three_dim=False)
        # Initialize temporary directory
        self.tmp_dir2D = tempfile.TemporaryDirectory(prefix="tmp.miscnn.")
        tmp_batches = os.path.join(self.tmp_dir2D.name, "batches")
        # Initialize Data IO
        self.data_io2D = Data_IO(io_interface2D,
                                 input_path=os.path.join(self.tmp_dir2D.name),
                                 output_path=os.path.join(self.tmp_dir2D.name),
                                 batch_path=tmp_batches, delete_batchDir=False)
        # Initialize Preprocessor
        self.pp2D = Preprocessor(self.data_io2D, batch_size=2,
                                 data_aug=None, analysis="fullimage")
        # Get sample list
        self.sample_list2D = self.data_io2D.get_indiceslist()
        # Create 3D imgaging and segmentation data set
        self.dataset3D = dict()
        for i in range(0, 6):
            img = np.random.rand(16, 16, 16) * 255
            self.img = img.astype(int)
            seg = np.random.rand(16, 16, 16) * 3
            self.seg = seg.astype(int)
            self.dataset3D["TEST.sample_" + str(i)] = (self.img, self.seg)
        # Initialize Dictionary IO Interface
        io_interface3D = Dictionary_interface(self.dataset3D, classes=3,
                                              three_dim=True)
        # Initialize temporary directory
        self.tmp_dir3D = tempfile.TemporaryDirectory(prefix="tmp.miscnn.")
        tmp_batches = os.path.join(self.tmp_dir3D.name, "batches")
        # Initialize Data IO
        self.data_io3D = Data_IO(io_interface3D,
                                 input_path=os.path.join(self.tmp_dir3D.name),
                                 output_path=os.path.join(self.tmp_dir3D.name),
                                 batch_path=tmp_batches, delete_batchDir=False)
        # Initialize Preprocessor
        self.pp3D = Preprocessor(self.data_io3D, batch_size=2,
                                 data_aug=None, analysis="fullimage")
        # Get sample list
        self.sample_list3D = self.data_io3D.get_indiceslist()

    # Delete all temporary files
    @classmethod
    def tearDownClass(self):
        self.tmp_dir2D.cleanup()
        self.tmp_dir3D.cleanup()

    #-------------------------------------------------#
    #                Base Functionality               #
    #-------------------------------------------------#
    # Class Creation
    def test_MODEL_create(self):
        nn2D = Neural_Network(preprocessor=self.pp2D)
        self.assertIsInstance(nn2D, Neural_Network)
        self.assertFalse(nn2D.three_dim)
        self.assertIsNotNone(nn2D.model)
        nn3D = Neural_Network(preprocessor=self.pp3D)
        self.assertIsInstance(nn3D, Neural_Network)
        self.assertTrue(nn3D.three_dim)
        self.assertIsNotNone(nn3D.model)

    # Model storage
    def test_MODEL_storage(self):
        nn = Neural_Network(preprocessor=self.pp3D)
        model_path = os.path.join(self.tmp_dir3D.name, "my_model.hdf5")
        nn.dump(model_path)
        self.assertTrue(os.path.exists(model_path))

    # Model loading
    def test_MODEL_loading(self):
        nn = Neural_Network(preprocessor=self.pp3D)
        model_path = os.path.join(self.tmp_dir3D.name, "my_model.hdf5")
        nn.dump(model_path)
        nn_new = Neural_Network(preprocessor=self.pp3D)
        nn_new.load(model_path)

    # Reseting weights
    def test_MODEL_resetWeights(self):
        nn = Neural_Network(preprocessor=self.pp3D)
        nn.reset_weights()

    #-------------------------------------------------#
    #                     Training                    #
    #-------------------------------------------------#
    def test_MODEL_training2D(self):
        nn = Neural_Network(preprocessor=self.pp2D)
        nn.train(self.sample_list2D, epochs=3)

    def test_MODEL_training3D(self):
        nn = Neural_Network(preprocessor=self.pp3D)
        nn.train(self.sample_list3D, epochs=3)

    #-------------------------------------------------#
    #                    Prediction                   #
    #-------------------------------------------------#
    def test_MODEL_prediction2D(self):
        nn = Neural_Network(preprocessor=self.pp2D)
        nn.predict(self.sample_list2D)
        for index in self.sample_list2D:
            sample = self.data_io2D.sample_loader(index, load_seg=True,
                                                  load_pred=True)
            self.assertIsNotNone(sample.pred_data)

    def test_MODEL_prediction3D(self):
        nn = Neural_Network(preprocessor=self.pp3D)
        nn.predict(self.sample_list3D)
        for index in self.sample_list3D:
            sample = self.data_io3D.sample_loader(index, load_seg=True,
                                                  load_pred=True)
            self.assertIsNotNone(sample.pred_data)

    def test_MODEL_prediction_returnOutput(self):
        nn = Neural_Network(preprocessor=self.pp2D)
        pred_list = nn.predict(self.sample_list2D, return_output=True)
        for pred in pred_list:
            self.assertIsNotNone(pred)
            self.assertEqual(pred.shape, (16,16))

    def test_MODEL_prediction_activationOutput(self):
        nn = Neural_Network(preprocessor=self.pp2D)
        pred_list = nn.predict(self.sample_list2D, return_output=True,
                               activation_output=True)
        for pred in pred_list:
            self.assertIsNotNone(pred)
            self.assertEqual(pred.shape, (16,16,3))

    #-------------------------------------------------#
    #                    Validation                   #
    #-------------------------------------------------#
    def test_MODEL_validation2D(self):
        nn = Neural_Network(preprocessor=self.pp2D)
        history = nn.evaluate(self.sample_list2D[0:4], self.sample_list2D[4:6],
                              epochs=3)
        self.assertIsNotNone(history)

    def test_MODEL_validation3D(self):
        nn = Neural_Network(preprocessor=self.pp3D)
        history = nn.evaluate(self.sample_list3D[0:4], self.sample_list3D[4:6],
                              epochs=3)
        self.assertIsNotNone(history)
