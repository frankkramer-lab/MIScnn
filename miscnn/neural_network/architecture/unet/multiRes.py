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
#                        Notes                        #
#-----------------------------------------------------#
# This architecture code was just copied from the official git repository which
# is mentioned in the MultiResUNet paper.
#
# Therefore, this code wasn't clean up, modified and just shallowly tested!!!
# Be aware when using it and please share you experiences and its performance
# with me.
#
# In the paper, it looks quite promising to me in comparison with the standard
# U-Net variant.
#
# A detailed performance analysis and code clean up of this architectures is on
# my to-do list.

#-----------------------------------------------------#
#               Source Code is based on:              #
#       https://github.com/nibtehaz/MultiResUNet      #
#                                                     #
#                     Reference:                      #
#          Nabil Ibtehaz and M. Sohel Rahman          #
#                  February 12, 2019                  #
#   MultiResUNet : Rethinking the U-Net Architecture  #
#    for Multimodal Biomedical Image Segmentation.    #
#      Neural Networks: Volume 121, January 2020,     #
#                     Pages 74-87                     #
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from tensorflow.keras.layers import Input, concatenate, BatchNormalization, Activation, add
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Conv3DTranspose
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import ELU, LeakyReLU
# Internal libraries/scripts
from miscnn.neural_network.architecture.abstract_architecture import Abstract_Architecture


#-----------------------------------------------------#
#         Architecture class: U-Net MultiRes          #
#-----------------------------------------------------#
""" The MultiRes variant of the popular U-Net architecture.

Methods:
    __init__                Object creation function
    create_model_2D:        Creating the 2D U-Net standard model using Keras
    create_model_3D:        Creating the 3D U-Net standard model using Keras
"""
class Architecture(Abstract_Architecture):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, activation='sigmoid'):
        # Parse parameter
        self.activation = activation

    #---------------------------------------------#
    #               Create 2D Model               #
    #---------------------------------------------#
    def create_model_2D(self, input_shape, n_labels=2):
        # Input layer
        inputs = Input(input_shape)

        mresblock1 = MultiResBlock_2D(32, inputs)
        pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
        mresblock1 = ResPath_2D(32, 4, mresblock1)

        mresblock2 = MultiResBlock_2D(32*2, pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
        mresblock2 = ResPath_2D(32*2, 3, mresblock2)

        mresblock3 = MultiResBlock_2D(32*4, pool2)
        pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
        mresblock3 = ResPath_2D(32*4, 2, mresblock3)

        mresblock4 = MultiResBlock_2D(32*8, pool3)
        pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
        mresblock4 = ResPath_2D(32*8, 1, mresblock4)

        mresblock5 = MultiResBlock_2D(32*16, pool4)

        up6 = concatenate([Conv2DTranspose(
            32*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
        mresblock6 = MultiResBlock_2D(32*8, up6)

        up7 = concatenate([Conv2DTranspose(
            32*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
        mresblock7 = MultiResBlock_2D(32*4, up7)

        up8 = concatenate([Conv2DTranspose(
            32*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
        mresblock8 = MultiResBlock_2D(32*2, up8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(
            2, 2), padding='same')(mresblock8), mresblock1], axis=3)
        mresblock9 = MultiResBlock_2D(32, up9)

        conv10 = conv2d_bn(mresblock9, n_labels, 1, 1, activation=self.activation)

        model = Model(inputs=[inputs], outputs=[conv10])
        return model


    #---------------------------------------------#
    #               Create 3D Model               #
    #---------------------------------------------#
    def create_model_3D(self, input_shape, n_labels=2):
        # Input layer
        inputs = Input(input_shape)

        # MultiResUNet 3D model code
        mresblock1 = MultiResBlock_3D(32, inputs)
        pool1 = MaxPooling3D(pool_size=(2, 2, 2))(mresblock1)
        mresblock1 = ResPath_3D(32, 4, mresblock1)

        mresblock2 = MultiResBlock_3D(32*2, pool1)
        pool2 = MaxPooling3D(pool_size=(2, 2, 2))(mresblock2)
        mresblock2 = ResPath_3D(32*2, 3,mresblock2)

        mresblock3 = MultiResBlock_3D(32*4, pool2)
        pool3 = MaxPooling3D(pool_size=(2, 2, 2))(mresblock3)
        mresblock3 = ResPath_3D(32*4, 2,mresblock3)

        mresblock4 = MultiResBlock_3D(32*8, pool3)
        pool4 = MaxPooling3D(pool_size=(2, 2, 2))(mresblock4)
        mresblock4 = ResPath_3D(32*8, 1,mresblock4)

        mresblock5 = MultiResBlock_3D(32*16, pool4)

        up6 = concatenate([Conv3DTranspose(32*8, (2, 2, 2), strides=(2, 2, 2), padding='same')(mresblock5), mresblock4], axis=4)
        mresblock6 = MultiResBlock_3D(32*8,up6)

        up7 = concatenate([Conv3DTranspose(32*4, (2, 2, 2), strides=(2, 2, 2), padding='same')(mresblock6), mresblock3], axis=4)
        mresblock7 = MultiResBlock_3D(32*4,up7)

        up8 = concatenate([Conv3DTranspose(32*2, (2, 2, 2), strides=(2, 2, 2), padding='same')(mresblock7), mresblock2], axis=4)
        mresblock8 = MultiResBlock_3D(32*2,up8)

        up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(mresblock8), mresblock1], axis=4)
        mresblock9 = MultiResBlock_3D(32,up9)

        conv10 = conv3d_bn(mresblock9 , n_labels, 1, 1, 1, activation=self.activation)

        model = Model(inputs=[inputs], outputs=[conv10])
        return model

#-----------------------------------------------------#
#             Subroutines for 3D version              #
#-----------------------------------------------------#
def conv3d_bn(x, filters, num_row, num_col, num_z, padding='same', strides=(1, 1, 1), activation='relu', name=None):
    '''
    3D Convolutional layers

    Arguments:
        x {keras layer} -- input layer
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
        num_z {int} -- length along z axis in filters
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})

    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv3D(filters, (num_row, num_col, num_z), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=4, scale=False)(x)

    if(activation==None):
        return x

    x = Activation(activation, name=name)(x)
    return x


def trans_conv3d_bn(x, filters, num_row, num_col, num_z, padding='same', strides=(2, 2, 2), name=None):
    '''
    2D Transposed Convolutional layers

    Arguments:
        x {keras layer} -- input layer
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
        num_z {int} -- length along z axis in filters

    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(2, 2, 2)})
        name {str} -- name of the layer (default: {None})

    Returns:
        [keras layer] -- [output layer]
    '''


    x = Conv3DTranspose(filters, (num_row, num_col, num_z), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=4, scale=False)(x)

    return x


def MultiResBlock_3D(U, inp, alpha = 1.67):
    '''
    MultiRes Block

    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer

    Returns:
        [keras layer] -- [output layer]
    '''

    W = alpha * U

    shortcut = inp

    shortcut = conv3d_bn(shortcut, int(W*0.167) + int(W*0.333) + int(W*0.5), 1, 1, 1, activation=None, padding='same')

    conv3x3 = conv3d_bn(inp, int(W*0.167), 3, 3, 3, activation='relu', padding='same')

    conv5x5 = conv3d_bn(conv3x3, int(W*0.333), 3, 3, 3, activation='relu', padding='same')

    conv7x7 = conv3d_bn(conv5x5, int(W*0.5), 3, 3, 3, activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=4)
    out = BatchNormalization(axis=4)(out)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=4)(out)

    return out

def ResPath_3D(filters, length, inp):
    '''
    ResPath

    Arguments:
        filters {int} -- [description]
        length {int} -- length of ResPath
        inp {keras layer} -- input layer

    Returns:
        [keras layer] -- [output layer]
    '''

    shortcut = inp
    shortcut = conv3d_bn(shortcut, filters , 1, 1, 1, activation=None, padding='same')

    out = conv3d_bn(inp, filters, 3, 3, 3, activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=4)(out)

    for i in range(length-1):

        shortcut = out
        shortcut = conv3d_bn(shortcut, filters , 1, 1, 1, activation=None, padding='same')

        out = conv3d_bn(out, filters, 3, 3, 3, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=4)(out)


    return out

#-----------------------------------------------------#
#             Subroutines for 2D version              #
#-----------------------------------------------------#
def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    '''
    2D Convolutional layers

    Arguments:
        x {keras layer} -- input layer
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters

    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})

    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if(activation == None):
        return x

    x = Activation(activation, name=name)(x)

    return x


def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
    '''
    2D Transposed Convolutional layers

    Arguments:
        x {keras layer} -- input layer
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters

    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(2, 2)})
        name {str} -- name of the layer (default: {None})

    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    return x


def MultiResBlock_2D(U, inp, alpha = 1.67):
    '''
    MultiRes Block

    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer

    Returns:
        [keras layer] -- [output layer]
    '''

    W = alpha * U

    shortcut = inp

    shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
                         int(W*0.5), 1, 1, activation=None, padding='same')

    conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3,
                        activation='relu', padding='same')

    conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3, 3,
                        activation='relu', padding='same')

    conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3, 3,
                        activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = BatchNormalization(axis=3)(out)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    return out


def ResPath_2D(filters, length, inp):
    '''
    ResPath

    Arguments:
        filters {int} -- [description]
        length {int} -- length of ResPath
        inp {keras layer} -- input layer

    Returns:
        [keras layer] -- [output layer]
    '''


    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, 1, 1,
                         activation=None, padding='same')

    out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    for i in range(length-1):

        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1,
                             activation=None, padding='same')

        out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

    return out
