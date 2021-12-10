#==============================================================================#
#  Author:       Philip Meyer                                                  #
#  Copyright:    2021 IT-Infrastructure for Translational Medical Research,    #
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
from miscnn.data_loading.interfaces import Dictionary_interface

from miscnn.multi_model.model import Model as BaseModel
from miscnn.multi_model.model_group import Model_Group
from miscnn import Data_IO, Preprocessor, Neural_Network


class ModelStub(BaseModel):

    def __init__(self, preprocessor):
        BaseModel.__init__(self, preprocessor)

        self.trained = False

    def train(self, sample_list, epochs=20, iterations=None, callbacks=[], class_weight=None):
        self.trained = True

    def predict(self, sample_list, activation_output=False):
        for sample in sample_list:
            s = self.preprocessor.data_io.sample_loader(sample)
            s.pred_data = np.zeros((16, 16, 16))
            self.preprocessor.data_io.save_prediction(s)
    # Evaluate the Model using the MIScnn pipeline
    def evaluate(self, training_samples, validation_samples, evaluation_path="evaluation", epochs=20, iterations=None, callbacks=[], store=True):
        self.trained = True
        for sample in validation_samples:
            s = self.preprocessor.data_io.sample_loader(sample)
            s.pred_data = np.zeros((16, 16, 16))
            self.preprocessor.data_io.save_prediction(s)

    def reset(self):
        self.trained = False

    def copy(self):
        stub = ModelStub(self.preprocessor)
        stub.trained = self.trained
        return stub

    # Dump model to file
    def dump(self, file_path):
        pass
    # Load model from file
    def load(self, file_path, custom_objects={}):
        pass


def drop (x, y):
    pass

#-----------------------------------------------------#
#              Unittest: Neural Network               #
#-----------------------------------------------------#
class ModelTEST(unittest.TestCase):
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

    # Delete all temporary files
    @classmethod
    def tearDownClass(self):
        self.tmp_dir2D.cleanup()

    #-------------------------------------------------#
    #                Base Functionality               #
    #-------------------------------------------------#
    # Class Creation
    def test_MODEL_create(self):
        group = Model_Group([ModelStub(self.pp2D), ModelStub(self.pp2D), ModelStub(self.pp2D)], self.pp2D, verify_preprocessor=True)
        self.assertIsInstance(group, Model_Group)


    # Model storage
    def test_MODEL_storage(self):
        group = Model_Group([ModelStub(self.pp2D), ModelStub(self.pp2D), ModelStub(self.pp2D)], self.pp2D, verify_preprocessor=True)
        model_path = os.path.join(self.tmp_dir2D.name, "model")
        group.dump(model_path)
        group_path = os.path.join(model_path, "group_" + str(group.id))
        self.assertTrue(os.path.exists(group_path))
        self.assertTrue(os.path.exists(os.path.join(group_path, "metadata.json")))

    # Model loading
    def test_MODEL_loading(self):
        group = Model_Group([ModelStub(self.pp2D), ModelStub(self.pp2D), ModelStub(self.pp2D)], self.pp2D, verify_preprocessor=True)
        model_path = os.path.join(self.tmp_dir2D.name, "model")
        group.dump(model_path)
        group.load(model_path)

    # Reseting weights
    def test_MODEL_resetWeights(self):
        group = Model_Group([ModelStub(self.pp2D), ModelStub(self.pp2D), ModelStub(self.pp2D)], self.pp2D, verify_preprocessor=True)
        group.reset()

    #-------------------------------------------------#
    #                     Training                    #
    #-------------------------------------------------#
    def test_MODEL_training(self):
        group = Model_Group([ModelStub(self.pp2D), ModelStub(self.pp2D), ModelStub(self.pp2D)], self.pp2D, verify_preprocessor=True)
        group.train(self.sample_list2D, epochs=3)

    #-------------------------------------------------#
    #                    Prediction                   #
    #-------------------------------------------------#

    def test_MODEL_prediction2D(self):
        group = Model_Group([ModelStub(self.pp2D), ModelStub(self.pp2D), ModelStub(self.pp2D)], self.pp2D, verify_preprocessor=True)
        group.predict(self.sample_list2D, drop)
        for index in self.sample_list2D:
            sample = self.data_io2D.sample_loader(index, load_seg=True,
                                                  load_pred=True)
            self.assertIsNotNone(sample.pred_data)

    #-------------------------------------------------#
    #                    Validation                   #
    #-------------------------------------------------#
    def test_MODEL_validation2D(self):
        group = Model_Group([ModelStub(self.pp2D), ModelStub(self.pp2D), ModelStub(self.pp2D)], self.pp2D, verify_preprocessor=True)
        history = group.evaluate(self.sample_list2D[0:4], self.sample_list2D[4:6],
                              epochs=3)
