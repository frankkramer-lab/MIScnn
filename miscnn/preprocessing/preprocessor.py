#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2019 IT-Infrastructure for Translational Medical Research,    #
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
# External libraries
from tqdm import tqdm
# Internal libraries/scripts
from miscnn.data_processing.data_augmentation import Data_Augmentation

#-----------------------------------------------------#
#                 Preprocessor class                  #
#-----------------------------------------------------#
# Class to handle all preprocessing functionalities
class Preprocessor:
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    """ Initialization function for creating a Preprocessor object.
    This class provides functionality for handling all preprocessing methods. This includes diverse
    optional preprocessing subfunctions like resampling, clipping, normalization or custom subfcuntions.
    This class processes the data into batches which are ready to be used for training, prediction and validation.

    The user is only required to create an instance of the Preprocessor class with the desired specifications
    and Data IO instance (optional also Data Augmentation instance).

    Args:
        data_io (Data_IO):                      Data IO class instance which handles all I/O operations according to the user
                                                defined interface
        data_aug (Data_Augmentation):           Data Augmentation class instance which performs diverse data augmentation techniques
        subfunctions (list of Subfunctions):    List of Subfunctions class instances which will be SEQUENTIALLY executed on the data set.
                                                (clipping, normalization, resampling, ...)
        batch_size (integer):                   Number of samples inside a single batch
        patchwise_analysis (boolean):           Boolean decision, if a full-image analysis or a patch-wise analysis should be performed
        patch_shape (integer tuple):            Size and shape of a patch. The variable has to be defined as a tuple.
                                                For Example: (64,128,128) for 64x128x128 patch cubes.
                                                Be aware that the x-axis represents the number of slices in 3D volumes.
                                                This parameter will be redundant if full image analysis is selected
                                                (patchwise_analysis is False)
        random_crop (boolean):                  In patchwise_analysis, patches will be randomly cropped from the image if this variable
                                                is True. If this parameter is false, the complete image will be sliced into a grid of
                                                patches.
                                                This parameter will be redundant if full image analysis is selected
                                                (patchwise_analysis is False)
        overlap (integer tuple):                In patchwise_analysis and without random cropping, an overlap can be defined between
                                                adjuncted patches.
                                                This parameter will be redundant if full image analysis is selected
                                                (patchwise_analysis is False).
                                                Only relevant if random_crop is False.
        skip_blanks (boolean):                  In patchwise_analysis and without random cropping, patches, containing only the
                                                background annotation, can be skipped with this option.
                                                This result into only training on relevant patches and ignore patches without any
                                                information.
                                                This parameter will be redundant if full image analysis is selected
                                                (patchwise_analysis is False).
                                                Only relevant if random_crop is False.
    """
    def __init__(self, data_io, batch_size, subfunctions=[],
                 data_aug=False, patchwise_analysis=True,
                 patch_shape=None, random_crop=False, overlap=(0,0,0),
                 skip_blanks=False):
        # Parse parameter
        self.data_io = data_io
        self.batch_size = batch_size
        self.subfunctions = subfunctions
        self.patchwise_analysis = patchwise_analysis
        self.patch_shape = patch_shape
        self.random_crop = random_crop
        self.overlap = overlap
        self.skip_blanks = skip_blanks
        # Create a default Data Augmentation instance if no one is provided
        if data_aug == False:
            self.data_augmentation = Data_Augmentation(data_io, 10, None)
        # Parse Data Augmentation
        else:
            self.data_augmentation = data_aug

    #---------------------------------------------#
    #              Run preprocessing              #
    #---------------------------------------------#
    def run(self, indices_list, training=True, validation=False):
        # Iterate over all samples
        for index in tqdm(indices_list):
            # Load sample
            sample = self.data_io.sample_loader(index, load_seg=training)
            # Run Subfunctions on the image data
            print(sample.img_data)
            for sf in self.subfunctions:
                sf.transform(sample, training=training)
            print(sample.img_data)
            # run subfunctions
            # for sf in subfunctions:
                # -> sf.run()

            # save image/seg as npz

            #if training
                # -> use data_aug ...
                # -> full image/patchwise
            #if prediction
                # -> NO data augmentation
                # -> grid or full image

        #return batchpointer?
        return None

    #---------------------------------------------#
    #             Patch-wise Analysis             #
    #---------------------------------------------#
    def patchwise_analysis(batch_size, patch_size, random=False,
                           overlap=(0,0,0), skip_blanks=False):
        if not random:
            print("1")
        else:
            print("2")

    #---------------------------------------------#
    #             Full-Image Analysis             #
    #---------------------------------------------#
    def fullimage_analysis(batch_size):
        print("lol")
