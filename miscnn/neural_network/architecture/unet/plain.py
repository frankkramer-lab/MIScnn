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
#                     Reference:                      #
#         Fabian Isensee, Klaus H. Maier-Hein.        #
#                     6 Aug 2019.                     #
#         An attempt at beating the 3D U-Net.         #
#           https://arxiv.org/abs/1908.02182          #
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Conv3DTranspose
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
# Internal libraries/scripts
from miscnn.neural_network.architecture.abstract_architecture import Abstract_Architecture

#-----------------------------------------------------#
#           Architecture class: U-Net Plain           #
#-----------------------------------------------------#
""" The Plain variant of the popular U-Net architecture.

Methods:
    __init__                Object creation function
    create_model_2D:        Creating the 2D U-Net plain model using Keras
    create_model_3D:        Creating the 3D U-Net plain model using Keras
"""
class Architecture(Abstract_Architecture):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, activation='softmax', batch_normalization=True):
        # Parse parameter
        self.activation = activation
        # Batch normalization settings
        self.ba_norm = batch_normalization
        # Create list of filters
        self.feature_map = [30, 60, 120, 240, 320]

    #---------------------------------------------#
    #               Create 2D Model               #
    #---------------------------------------------#
    def create_model_2D(self, input_shape, n_labels=2):
        # Input layer
        inputs = Input(input_shape)
        # Start the CNN Model chain with adding the inputs as first tensor
        cnn_chain = inputs
        # Cache contracting normalized conv layers
        # for later copy & concatenate links
        contracting_convs = []

        # Contracting layers
        for i in range(0, len(self.feature_map)):
            neurons = self.feature_map[i]
            cnn_chain = conv_layer_2D(cnn_chain, neurons, self.ba_norm, strides=1)
            cnn_chain = conv_layer_2D(cnn_chain, neurons, self.ba_norm, strides=1)
            contracting_convs.append(cnn_chain)
            cnn_chain = MaxPooling2D(pool_size=(2, 2))(cnn_chain)

        # Middle Layer
        neurons = self.feature_map[-1]
        cnn_chain = conv_layer_2D(cnn_chain, neurons, self.ba_norm, strides=1)
        cnn_chain = conv_layer_2D(cnn_chain, neurons, self.ba_norm, strides=1)

        # Expanding Layers
        for i in reversed(range(0, len(self.feature_map))):
            neurons = self.feature_map[i]
            cnn_chain = Conv2DTranspose(neurons, (2, 2), strides=(2, 2),
                                        padding='same')(cnn_chain)
            cnn_chain = concatenate([cnn_chain, contracting_convs[i]], axis=-1)
            cnn_chain = conv_layer_2D(cnn_chain, neurons, self.ba_norm, strides=1)
            cnn_chain = conv_layer_2D(cnn_chain, neurons, self.ba_norm, strides=1)

        # Output Layer
        conv_out = Conv2D(n_labels, (1, 1), activation=self.activation)(cnn_chain)
        # Create Model with associated input and output layers
        model = Model(inputs=[inputs], outputs=[conv_out])
        # Return model
        return model

    #---------------------------------------------#
    #               Create 3D Model               #
    #---------------------------------------------#
    def create_model_3D(self, input_shape, n_labels=2):
        # Input layer
        inputs = Input(input_shape)
        # Start the CNN Model chain with adding the inputs as first tensor
        cnn_chain = inputs
        # Cache contracting normalized conv layers
        # for later copy & concatenate links
        contracting_convs = []

        # First contracting layer
        neurons = self.feature_map[0]
        cnn_chain = conv_layer_3D(cnn_chain, neurons, self.ba_norm, strides=1)
        cnn_chain = conv_layer_3D(cnn_chain, neurons, self.ba_norm, strides=1)
        contracting_convs.append(cnn_chain)
        cnn_chain = MaxPooling3D(pool_size=(1, 2, 2))(cnn_chain)

        # Remaining contracting layers
        for i in range(1, len(self.feature_map)):
            neurons = self.feature_map[i]
            cnn_chain = conv_layer_3D(cnn_chain, neurons, self.ba_norm, strides=1)
            cnn_chain = conv_layer_3D(cnn_chain, neurons, self.ba_norm, strides=1)
            contracting_convs.append(cnn_chain)
            cnn_chain = MaxPooling3D(pool_size=(2, 2, 2))(cnn_chain)

        # Middle Layer
        neurons = self.feature_map[-1]
        cnn_chain = conv_layer_3D(cnn_chain, neurons, self.ba_norm, strides=1)
        cnn_chain = conv_layer_3D(cnn_chain, neurons, self.ba_norm, strides=1)

        # Expanding Layers except last layer
        for i in reversed(range(1, len(self.feature_map))):
            neurons = self.feature_map[i]
            cnn_chain = Conv3DTranspose(neurons, (2, 2, 2), strides=(2, 2, 2),
                                        padding='same')(cnn_chain)
            cnn_chain = concatenate([cnn_chain, contracting_convs[i]], axis=-1)
            cnn_chain = conv_layer_3D(cnn_chain, neurons, self.ba_norm, strides=1)
            cnn_chain = conv_layer_3D(cnn_chain, neurons, self.ba_norm, strides=1)

        # Last expanding layer
        neurons = self.feature_map[0]
        cnn_chain = Conv3DTranspose(neurons, (1, 2, 2), strides=(1, 2, 2),
                                    padding='same')(cnn_chain)
        cnn_chain = concatenate([cnn_chain, contracting_convs[0]], axis=-1)
        cnn_chain = conv_layer_3D(cnn_chain, neurons, self.ba_norm, strides=1)
        cnn_chain = conv_layer_3D(cnn_chain, neurons, self.ba_norm, strides=1)

        # Output Layer
        conv_out = Conv3D(n_labels, (1, 1, 1), activation=self.activation)(cnn_chain)
        # Create Model with associated input and output layers
        model = Model(inputs=[inputs], outputs=[conv_out])
        # Return model
        return model

#-----------------------------------------------------#
#                   Subroutines 2D                    #
#-----------------------------------------------------#
# Convolution layer
def conv_layer_2D(input, neurons, ba_norm, strides=1):
    conv = Conv2D(neurons, (3,3), activation='relu', padding='same',
                  strides=strides)(input)
    if ba_norm : conv = BatchNormalization(momentum=0.99)(conv)
    return conv

#-----------------------------------------------------#
#                   Subroutines 3D                    #
#-----------------------------------------------------#
# Convolution layer
def conv_layer_3D(input, neurons, ba_norm, strides=1):
    conv = Conv3D(neurons, (3,3,3), activation='relu', padding='same',
                  strides=strides)(input)
    if ba_norm : conv = BatchNormalization(momentum=0.99)(conv)
    return conv
