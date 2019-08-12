#==============================================================================#
# Author:       Dominik Müller                                                 #
# Copyright:    2019 IT-Infrastructure for Translational Medical Research,     #
#               University of Augsburg                                         #
# License:      GNU General Public License v3.0                                #
#                                                                              #
# Unless required by applicable law or agreed to in writing, software          #
# distributed under the License is distributed on an "AS IS" BASIS,            #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.     #
# See the License for the specific language governing permissions and          #
# limitations under the License.                                               #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from abc import ABC, abstractmethod

#-----------------------------------------------------#
#       Abstract Interface for the Data IO class      #
#-----------------------------------------------------#
""" An abstract base class for a Data_IO interface.

Methods:
    initialize:             Prepare the data set and create indices list
    load_image:             Load an image
    load_segmentation:      Load a segmentation
    load_prediction:        Load a prediction from file
    save_prediction:        Save a prediction to file
"""
class Abstract_IO(ABC):
    #---------------------------------------------#
    #                  initialize                 #
    #---------------------------------------------#
    """ Initialize and prepare the image data set, return the number of samples in the data set

        Parameter:
            input_path (string):    Path to the input data directory, in which all imaging data have to be accessible
        Return:
            indices_list [list]:    List of indices. The Data_IO class will iterate over this list and
                                    call the load_image and load_segmentation functions providing the current index.
                                    This can be used to train/predict on just a subset of the data set.
                                    e.g. indices_list = [0,1,9]
                                    -> load_image(0) | load_image(1) | load_image(9)
    """
    @abstractmethod
    def initialize(self, input_path):
        pass
    #---------------------------------------------#
    #                  load_image                 #
    #---------------------------------------------#
    """ Load the image with the index i from the data set and return it as a numpy matrix.

        Parameter:
            index (variable):       An index from the provided indices_list of the initialize function
        Return:
            image [numpy matrix]:   A numpy matrix/array containing the image
    """
    @abstractmethod
    def load_image(self, i):
        pass
    #---------------------------------------------#
    #              load_segmentation              #
    #---------------------------------------------#
    """ Load the segmentation of the image with the index i from the data set and return it as a numpy matrix.

        Parameter:
            index (variable):       An index from the provided indices_list of the initialize function
        Return:
            seg [numpy matrix]:     A numpy matrix/array containing the segmentation
    """
    @abstractmethod
    def load_segmentation(self, i):
        pass
    #---------------------------------------------#
    #               load_prediction               #
    #---------------------------------------------#
    """ Load the prediction of the image with the index i from the output directory
        and return it as a numpy matrix.

        Parameter:
            index (variable):       An index from the provided indices_list of the initialize function
            output_path (string):   Path to the output directory in which MIScnn predictions are stored.
        Return:
            pred [numpy matrix]:    A numpy matrix/array containing the prediction
    """
    @abstractmethod
    def load_prediction(self, i, output_path):
        pass
    #---------------------------------------------#
    #               save_prediction               #
    #---------------------------------------------#
    """ Backup the prediction of the image with the index i into the output directory.

        Parameter:
            pred (numpy matrix):    MIScnn computed prediction for the sample index
            index (variable):       An index from the provided indices_list of the initialize function
            output_path (string):   Path to the output directory in which MIScnn predictions are stored.
                                    This directory will be created if not existent
        Return:
            None
    """
    @abstractmethod
    def save_prediction(self, pred, i, output_path):
        pass
