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
import numpy
import math

#-----------------------------------------------------#
#                 Image Sample - class                #
#-----------------------------------------------------#
# Object containing an image and the associated segmentation
class Sample:
    # Initialize class variable
    index = None
    img_data = None
    seg_data = None
    pred_data = None
    shape = None
    channels = None

    # Create a Sample object
    def __init__(self, index, image, channels):
        # Preprocess image data if required
        if image.shape[-1] != channels:
            image = numpy.reshape(image, image.shape + (channels,))
        # Cache data
        self.index = index
        self.img_data = image
        self.channels = channels
        self.shape = self.img_data.shape

    # Add and preprocess a segmentation annotation
    def add_segmentation(self, seg):
        if seg.shape[-1] != 1:
            seg = numpy.reshape(seg, seg.shape + (1,))
        self.seg_data = seg

    # Add and preprocess a prediction annotation
    def add_prediction(self, pred):
        if pred.shape[-1] != 1:
            pred = numpy.reshape(pred, pred.shape + (1,))
        self.pred_data = pred
