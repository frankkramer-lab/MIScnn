import tensorflow_addons as tfa

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Conv3DTranspose
from tensorflow.keras.layers import SpatialDropout2D, SpatialDropout3D, LeakyReLU, LayerNormalization
# Internal libraries/scripts
from miscnn.neural_network.architecture.abstract_architecture import Abstract_Architecture

# -----------------------------------------------------#
#           Architecture class: U-Net Plain           #
# -----------------------------------------------------#
""" The Plain variant of the popular U-Net architecture.
Methods:
    __init__                Object creation function
    create_model_2D:        Creating the 2D U-Net plain model using Keras
    create_model_3D:        Creating the 3D U-Net plain model using Keras
"""


class Architecture(Abstract_Architecture):
    # ---------------------------------------------#
    #                Initialization               #
    # ---------------------------------------------#
    def __init__(self, activation='softmax', conv_layer_activation='lrelu',
                 instance_normalization=True, instance_normalization_params=None,
                 layer_normalization_params=None, dropout=0.5, feature_maps=None):
        # Parse parameter
        if instance_normalization_params is None:
            instance_normalization_params = {'epsilon': 1e-5}
        if layer_normalization_params is None:
            layer_normalization_params = {'epsilon': 1e-5}
        self.activation = activation
        # Parse activation layer
        if conv_layer_activation == "lrelu":
            self.conv_layer_activation = LeakyReLU(alpha=0.1)
        # Batch normalization settings
        self.inst_norm = instance_normalization
        self.inst_norm_params = instance_normalization_params
        self.layer_norm_params = layer_normalization_params
        # Dropout params
        self.dropout = dropout
        # Create list of filters
        if feature_maps is None:
            self.feature_maps = {
                4: [40, 240],
                2: [40, 80, 160, 220]
            }
        else:
            self.feature_maps = feature_maps

    # ---------------------------------------------#
    #               Create 2D Model               #
    # ---------------------------------------------#
    def create_model_2D(self, input_shape, n_labels=2):
        # Input layer
        inputs = Input(input_shape)
        # Cache contracting normalized conv layers
        # for later copy & concatenate links
        contracting_convs = {c: [] for c in self.feature_maps.keys()}
        all_middle_chains = []
        all_chains = []

        for stride, feature_map in self.feature_maps.items():
            # Start the CNN Model chain with adding the inputs as first tensor
            cnn_chain = inputs

            # Contracting layers
            for i in range(0, len(feature_map)):
                neurons = feature_map[i]
                cnn_chain = conv_layer_2D(cnn_chain, neurons, self.conv_layer_activation,
                                          self.inst_norm, self.inst_norm_params)
                cnn_chain = conv_layer_2D(cnn_chain, neurons, self.conv_layer_activation,
                                          self.inst_norm, self.inst_norm_params)
                cnn_chain = SpatialDropout2D(self.dropout)(cnn_chain)
                contracting_convs[stride].append(cnn_chain)
                cnn_chain = MaxPooling2D(pool_size=(stride, stride))(cnn_chain)

            # Middle Layer
            neurons = feature_map[-1]
            cnn_chain = conv_layer_2D(cnn_chain, neurons, self.conv_layer_activation,
                                      self.inst_norm, self.inst_norm_params)
            cnn_chain = conv_layer_2D(cnn_chain, neurons, self.conv_layer_activation,
                                      self.inst_norm, self.inst_norm_params)
            all_middle_chains.append(cnn_chain)

        cnn_chain1 = concatenate(all_middle_chains)
        cnn_chain1 = LayerNormalization(**self.layer_norm_params)(cnn_chain1)

        for stride, feature_map in self.feature_maps.items():
            # Start the CNN Model chain with adding the inputs as first tensor
            cnn_chain = cnn_chain1
            # Expanding Layers
            for i in reversed(range(0, len(feature_map))):
                neurons = feature_map[i]
                cnn_chain = Conv2DTranspose(neurons, (stride, stride),
                                            strides=(stride, stride),
                                            padding='same')(cnn_chain)
                cnn_chain = SpatialDropout2D(self.dropout)(cnn_chain)
                cnn_chain = concatenate([cnn_chain, contracting_convs[stride][i]], axis=-1)
                cnn_chain = conv_layer_2D(cnn_chain, neurons, self.conv_layer_activation,
                                          self.inst_norm, self.inst_norm_params)
                cnn_chain = conv_layer_2D(cnn_chain, neurons, self.conv_layer_activation,
                                          self.inst_norm, self.inst_norm_params)

            all_chains.append(cnn_chain)

        # Output Layer
        conv_out = Conv2D(n_labels, (1, 1), activation=self.activation)(concatenate(all_chains, axis=-1))
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
        # Cache contracting normalized conv layers
        # for later copy & concatenate links
        contracting_convs = {c: [] for c in self.feature_maps.keys()}
        all_middle_chains = []
        all_chains = []

        for stride, feature_map in self.feature_maps.items():
            # Start the CNN Model chain with adding the inputs as first tensor
            cnn_chain = inputs

            # Contracting layers
            for i in range(0, len(feature_map)):
                neurons = feature_map[i]
                cnn_chain = conv_layer_3D(cnn_chain, neurons, self.conv_layer_activation,
                                          self.inst_norm, self.inst_norm_params)
                cnn_chain = conv_layer_3D(cnn_chain, neurons, self.conv_layer_activation,
                                          self.inst_norm, self.inst_norm_params)
                cnn_chain = SpatialDropout3D(self.dropout)(cnn_chain)
                contracting_convs[stride].append(cnn_chain)
                cnn_chain = MaxPooling3D(pool_size=(stride, stride, stride))(cnn_chain)

            # Middle Layer
            neurons = feature_map[-1]
            cnn_chain = conv_layer_3D(cnn_chain, neurons, self.conv_layer_activation,
                                      self.inst_norm, self.inst_norm_params)
            cnn_chain = conv_layer_3D(cnn_chain, neurons, self.conv_layer_activation,
                                      self.inst_norm, self.inst_norm_params)
            all_middle_chains.append(cnn_chain)

        cnn_chain1 = concatenate(all_middle_chains)
        cnn_chain1 = LayerNormalization(**self.layer_norm_params)(cnn_chain1)

        for stride, feature_map in self.feature_maps.items():
            # Start the CNN Model chain with adding the inputs as first tensor
            cnn_chain = cnn_chain1
            # Expanding Layers
            for i in reversed(range(0, len(feature_map))):
                neurons = feature_map[i]
                cnn_chain = Conv3DTranspose(neurons, (stride, stride, stride),
                                            strides=(stride, stride, stride),
                                            padding='same')(cnn_chain)
                cnn_chain = SpatialDropout3D(self.dropout)(cnn_chain)
                cnn_chain = concatenate([cnn_chain, contracting_convs[stride][i]], axis=-1)
                cnn_chain = conv_layer_3D(cnn_chain, neurons, self.conv_layer_activation,
                                          self.inst_norm, self.inst_norm_params)
                cnn_chain = conv_layer_3D(cnn_chain, neurons, self.conv_layer_activation,
                                          self.inst_norm, self.inst_norm_params)

            all_chains.append(cnn_chain)

        # Output Layer
        conv_out = Conv3D(n_labels, (1, 1, 1), activation=self.activation)(concatenate(all_chains, axis=-1))
        # Create Model with associated input and output layers
        model = Model(inputs=[inputs], outputs=[conv_out])
        # Return model
        return model


# -----------------------------------------------------#
#                   Subroutines 2D                    #
# -----------------------------------------------------#
# Convolution layer
def conv_layer_2D(input, neurons, activation, inst_norm, inst_norm_params, strides=1):
    conv = Conv2D(neurons, (3, 3), padding='same', strides=strides)(input)
    if inst_norm:
        conv = tfa.layers.InstanceNormalization(**inst_norm_params)(conv)

    return Activation(activation)(conv)

#-----------------------------------------------------#
#                   Subroutines 3D                    #
#-----------------------------------------------------#
# Convolution layer
def conv_layer_3D(input, neurons, activation, inst_norm, inst_norm_params, strides=1):
    conv = Conv3D(neurons, (3, 3, 3), padding='same', strides=strides)(input)

    if inst_norm:
        conv = tfa.layers.InstanceNormalization(**inst_norm_params)(conv)

    return Activation(activation)(conv)
