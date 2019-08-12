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
#External libraries
import numpy
import math

#-----------------------------------------------------#
#                 Image Sample - class                #
#-----------------------------------------------------#
# Object containing an image and the associated segmentation
class Sample:
    # Initialize class variable
    vol_data = None
    seg_data = None
    shape = None
    channels = None

    # Create a Sample object
    def __init__(self, volume, channels):
        # Add and preprocess volume data
        vol_data = volume.get_data()
        self.vol_data = numpy.reshape(vol_data, vol_data.shape + (channels,))
        self.channels = channels
        self.shape = self.vol_data.shape

    # Add and preprocess segmentation annotation
    def add_segmentation(self, segmentation):
        seg_data = segmentation.get_data()
        self.seg_data = numpy.reshape(seg_data, seg_data.shape + (1,))
