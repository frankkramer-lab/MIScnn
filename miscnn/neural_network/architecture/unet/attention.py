# ==============================================================================#
#  Author:       Dennis Hartmann                                               #
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
# ==============================================================================#
# -----------------------------------------------------#
#                     Reference:                       #
#                  Ozan Oktay et al.                   #
#                    11 April 2018.                    #
#          Attention U-Net: Learning Where             #
#             to Look for the Pancreas                 #
#                    MIDL'18.                          #
# -----------------------------------------------------#
#                   Library imports                    #
# -----------------------------------------------------#
# External libraries
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Activation, add, Lambda, multiply
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Conv3DTranspose, UpSampling3D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as k
# Internal libraries/scripts
from miscnn.neural_network.architecture.abstract_architecture import Abstract_Architecture

# -----------------------------------------------------#
#         Architecture class: Attention U-Net          #
# -----------------------------------------------------#
""" The Standard variant of the popular U-Net architecture.

Methods:
    __init__                Object creation function
    create_model_2D:        Creating the 2D Attention U-Net standard model using Keras
    create_model_3D:        Creating the 3D Attention U-Net standard model using Keras
"""


class Architecture(Abstract_Architecture):
    # ---------------------------------------------#
    #                Initialization               #
    # ---------------------------------------------#
    def __init__(self, n_filters=32, depth=4, activation='softmax',
                 batch_normalization=True):
        # Parse parameter
        self.n_filters = n_filters
        self.depth = depth
        self.activation = activation
        # Batch normalization settings
        self.ba_norm = batch_normalization
        self.ba_norm_momentum = 0.99

    # ---------------------------------------------#
    #               Create 2D Model               #
    # ---------------------------------------------#
    def create_model_2D(self, input_shape, n_labels=2):
        # Input layer
        inputs = Input(input_shape)
        # Start the CNN Model chain with adding the inputs as first tensor
        cnn_chain = inputs
        # Cache contracting normalized conv layers
        # for later copy & concatenate links
        contracting_convs = []

        # Contracting Layers
        for i in range(0, self.depth):
            neurons = self.n_filters * 2 ** i
            cnn_chain, last_conv = contracting_layer_2D(cnn_chain, neurons,
                                                        self.ba_norm,
                                                        self.ba_norm_momentum)
            contracting_convs.append(last_conv)

        # Middle Layer
        neurons = self.n_filters * 2 ** self.depth
        cnn_chain = middle_layer_2D(cnn_chain, neurons, self.ba_norm,
                                    self.ba_norm_momentum)

        # Expanding Layers
        for i in reversed(range(0, self.depth)):
            neurons = self.n_filters * 2 ** i
            cnn_chain = expanding_layer_2D(cnn_chain, neurons,
                                           contracting_convs[i], self.ba_norm,
                                           self.ba_norm_momentum)

        # Output Layer
        conv_out = Conv2D(n_labels, (1, 1),
                          activation=self.activation)(cnn_chain)
        # Create Model with associated input and output layers
        model = Model(inputs=[inputs], outputs=[conv_out])
        # Return model
        return model

    # ---------------------------------------------#
    #               Create 3D Model               #
    # ---------------------------------------------#
    def create_model_3D(self, input_shape, n_labels=2):
        # Input layer
        inputs = Input(input_shape)
        # Start the CNN Model chain with adding the inputs as first tensor
        cnn_chain = inputs
        # Cache contracting normalized conv layers
        # for later copy & concatenate links
        contracting_convs = []

        # Contracting Layers
        for i in range(0, self.depth):
            neurons = self.n_filters * 2 ** i
            cnn_chain, last_conv = contracting_layer_3D(cnn_chain, neurons,
                                                        self.ba_norm,
                                                        self.ba_norm_momentum)
            contracting_convs.append(last_conv)

        # Middle Layer
        neurons = self.n_filters * 2 ** self.depth
        cnn_chain = middle_layer_3D(cnn_chain, neurons, self.ba_norm,
                                    self.ba_norm_momentum)

        # Expanding Layers
        for i in reversed(range(0, self.depth)):
            neurons = self.n_filters * 2 ** i
            cnn_chain = expanding_layer_3D(cnn_chain, neurons,
                                           contracting_convs[i], self.ba_norm,
                                           self.ba_norm_momentum)

        # Output Layer
        conv_out = Conv3D(n_labels, (1, 1, 1),
                          activation=self.activation)(cnn_chain)
        # Create Model with associated input and output layers
        model = Model(inputs=[inputs], outputs=[conv_out])
        # Return model
        return model


# -----------------------------------------------------#
#                   Subroutines all                    #
# -----------------------------------------------------#
def repeat_elem(tensor, rep, axs=3):
    # lambda function to repeat Repeats the elements of a tensor along an axis
    # by a factor of rep.
    # If tensor has shape (None, 256,256,3), lambda will return a tensor of shape
    # (None, 256,256,6), if specified axis=3 and rep=2.

    return Lambda(lambda x, repnum: k.repeat_elements(x, repnum, axis=axs),
                  arguments={'repnum': rep})(tensor)


# -----------------------------------------------------#
#                   Subroutines 2D                    #
# -----------------------------------------------------#
def gating_signal2D(input, out_size, batch_norm=False):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    x = Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def attention_block2D(x, gating, inter_shape):
    shape_x = k.int_shape(x)

    # Getting the x signal to the same shape as the gating signal
    theta_x = Conv2D(filters=inter_shape, kernel_size=3, strides=2, padding='same')(x)

    # Getting the gating signal to the same number of filters as the inter_shape
    phi_g = Conv2D(filters=inter_shape, kernel_size=1, strides=1, padding='same')(gating)

    concat_xg = add([phi_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(filters=1, kernel_size=1, padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    upsample_psi = UpSampling2D(size=2)(sigmoid_xg)

    upsample_psi = repeat_elem(upsample_psi, shape_x[3])

    y = multiply([upsample_psi, x])

    # Final 1x1 convolution to consolidate attention signal to original x dimensions
    result = Conv2D(filters=shape_x[3], kernel_size=1, strides=1, padding='same')(y)
    result_bn = BatchNormalization()(result)
    return result_bn


# Create a contracting layer
def contracting_layer_2D(input, neurons, ba_norm, ba_norm_momentum):
    conv1 = Conv2D(filters=neurons, kernel_size=3, activation='relu', padding='same')(input)
    if ba_norm: conv1 = BatchNormalization(momentum=ba_norm_momentum)(conv1)
    conv2 = Conv2D(filters=neurons, kernel_size=3, activation='relu', padding='same')(conv1)
    if ba_norm: conv2 = BatchNormalization(momentum=ba_norm_momentum)(conv2)
    pool = MaxPooling2D(pool_size=2)(conv2)
    return pool, conv2


# Create the middle layer between the contracting and expanding layers
def middle_layer_2D(input, neurons, ba_norm, ba_norm_momentum):
    conv_m1 = Conv2D(filters=neurons, kernel_size=3, activation='relu', padding='same')(input)
    if ba_norm: conv_m1 = BatchNormalization(momentum=ba_norm_momentum)(conv_m1)
    conv_m2 = Conv2D(filters=neurons, kernel_size=3, activation='relu', padding='same')(conv_m1)
    if ba_norm: conv_m2 = BatchNormalization(momentum=ba_norm_momentum)(conv_m2)
    return conv_m2


# Create an expanding layer
def expanding_layer_2D(input, neurons, concatenate_link, ba_norm,
                       ba_norm_momentum):
    gating = gating_signal2D(input, neurons, ba_norm)
    att = attention_block2D(concatenate_link, gating, neurons)
    up = concatenate([Conv2DTranspose(filters=neurons, kernel_size=2, strides=2,
                                      padding='same')(input), att], axis=-1)
    conv1 = Conv2D(filters=neurons, kernel_size=3, activation='relu', padding='same')(up)
    if ba_norm: conv1 = BatchNormalization(momentum=ba_norm_momentum)(conv1)
    conv2 = Conv2D(filters=neurons, kernel_size=3, activation='relu', padding='same')(conv1)
    if ba_norm: conv2 = BatchNormalization(momentum=ba_norm_momentum)(conv2)
    return conv2


# -----------------------------------------------------#
#                   Subroutines 3D                    #
# -----------------------------------------------------#
def gating_signal3D(input, out_size, batch_norm=False):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    x = Conv3D(out_size, kernel_size=1, padding='same')(input)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def attention_block3D(x, gating, inter_shape):
    shape_x = k.int_shape(x)

    # Getting the x signal to the same shape as the gating signal
    theta_x = Conv3D(filters=inter_shape, kernel_size=3, strides=2, padding='same')(x)  # 16

    # Getting the gating signal to the same number of filters as the inter_shape
    phi_g = Conv3D(filters=inter_shape, kernel_size=1, strides=1, padding='same')(gating)

    concat_xg = add([phi_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv3D(filters=1, kernel_size=1, padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    upsample_psi = UpSampling3D(size=2)(sigmoid_xg)

    upsample_psi = repeat_elem(upsample_psi, shape_x[4], axs=4)

    y = multiply([upsample_psi, x])

    result = Conv3D(filters=shape_x[4], kernel_size=1, strides=1, padding='same')(y)
    result_bn = BatchNormalization()(result)
    return result_bn


# Create a contracting layer
def contracting_layer_3D(input, neurons, ba_norm, ba_norm_momentum):
    conv1 = Conv3D(filters=neurons, kernel_size=3, activation='relu', padding='same')(input)
    if ba_norm: conv1 = BatchNormalization(momentum=ba_norm_momentum)(conv1)
    conv2 = Conv3D(filters=neurons, kernel_size=3, activation='relu', padding='same')(conv1)
    if ba_norm: conv2 = BatchNormalization(momentum=ba_norm_momentum)(conv2)
    pool = MaxPooling3D(pool_size=2)(conv2)
    return pool, conv2


# Create the middle layer between the contracting and expanding layers
def middle_layer_3D(input, neurons, ba_norm, ba_norm_momentum):
    conv_m1 = Conv3D(filters=neurons, kernel_size=3, activation='relu', padding='same')(input)
    if ba_norm: conv_m1 = BatchNormalization(momentum=ba_norm_momentum)(conv_m1)
    conv_m2 = Conv3D(filters=neurons, kernel_size=3, activation='relu', padding='same')(conv_m1)
    if ba_norm: conv_m2 = BatchNormalization(momentum=ba_norm_momentum)(conv_m2)
    return conv_m2


# Create an expanding layer
def expanding_layer_3D(input, neurons, concatenate_link, ba_norm,
                       ba_norm_momentum):
    gating = gating_signal3D(input, neurons, ba_norm)
    att = attention_block3D(concatenate_link, gating, neurons)  # Neurons = Filter?
    up = concatenate([Conv3DTranspose(filters=neurons, kernel_size=2, strides=2,
                                      padding='same')(input), att], axis=-1)
    conv1 = Conv3D(filters=neurons, kernel_size=3, activation='relu', padding='same')(up)
    if ba_norm: conv1 = BatchNormalization(momentum=ba_norm_momentum)(conv1)
    conv2 = Conv3D(filters=neurons, kernel_size=3, activation='relu', padding='same')(conv1)
    if ba_norm: conv2 = BatchNormalization(momentum=ba_norm_momentum)(conv2)
    return conv2
