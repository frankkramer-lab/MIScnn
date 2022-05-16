#==============================================================================#
#  Author:       Philip Meyer, Dennis Hartmann                                 #
#  Copyright:    2022 IT-Infrastructure for Translational Medical Research,    #
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
import numpy as np
from miscnn.utils.visualizer import visualize_samples


'''
The certainty-map is the visualization of the softmax-output of the last layer of the unet.
@type  sample_list: List of Strings
@param sample_list: Names of the image-files
@type  model: miscnn-neural_network.mode.Neural_Network
@param model: Model of the neural network of MISCNN
@type  cls: List of Strings or Integers
@param cls: Names of the classes
@type  out_dir: String
@param out_dir: Path where to save the maps
@type  progress: Boolean
@param progress: Determines whether a progress bar should be displayed
@type  return_data: Boolean
@param return_data: Determines whether the map should be returned
@return: Nothing if return_data==False, Certaintymap if return_data==True
'''
def compute_certaintymap(sample_list, model, cls=[0, 1, 2, 3, 4], out_dir="vis", progress=False, return_data=False):
    pp = model.preprocessor
    skip_blanks = pp.patchwise_skip_blanks
    pp.patchwise_skip_blanks = False
    pp.prepare_batches = False
    pp.prepare_subfunctions = False
    preprocessed_input = [pp.run([s], training=False) for s in sample_list]
    pp.patchwise_skip_blanks = skip_blanks
    preprocessed_input = [[a[0] for a in s] for s in preprocessed_input]

    time_addition = 2

    if return_data:
        time_addition = 0

    if progress:
        t = [len(s) + time_addition for s in preprocessed_input]
        patches = sum(t)
        pbar = tqdm(total=patches)

    if return_data:
        ret = []

    for index, sample in zip(sample_list, preprocessed_input):
        should_maintain = ("shape_" + str(index)) in pp.cache.keys()

        if should_maintain:
            pre_shp = pp.cache["shape_" + str(index)]

        pred = np.asarray(model.predict([index], return_output=True, activation_output=True))

        if should_maintain:
            pp.cache["shape_" + str(index)] = pre_shp

        if progress:
            pbar.update(1)
        temp = []
        for c in range(pred.shape[-1]):
            if not return_data:
                s = pp.data_io.sample_loader(index, False, False)
                s.index = s.index + ".certainty"
                s.pred_data = pred[..., c]
                visualize_samples([s], out_dir=out_dir + str(cls[c]), mask_pred=True, colorbar=True)
                if progress:
                    pbar.update(1)
            else:
                temp.append(pred[..., c])
        if return_data:
            ret.append(temp)

    if return_data:
        return ret
