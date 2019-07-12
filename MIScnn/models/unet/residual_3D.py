#-----------------------------------------------------#
#               Source Code is based on:              #
# https://github.com/mrkolarik/3D-brain-segmentation  #
#                                                     #
#                     Reference:                      #
#Kolařík, M., Burget, R., Uher, V., Říha, K., & Dutta,#
#                    M. K. (2019).                    #
#  Optimized High Resolution 3D Dense-U-Net Network   #
#          for Brain and Spine Segmentation.          #
#        Applied Sciences, 9(3), vol. 9, no. 3.       #
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
from keras.models import Model
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose

#-----------------------------------------------------#
#                    Core Function                    #
#-----------------------------------------------------#
def Unet(input_shape, n_labels, n_filters=32, depth=4, activation='sigmoid'):
    # Input layer
    inputs = Input(input_shape)
    # Start the CNN Model chain with adding the inputs as first tensor
    cnn_chain = inputs
    # Cache contracting normalized conv layers
    # for later copy & concatenate links
    contracting_convs = []

    # Contracting Layers
    for i in range(0, depth):
        neurons = n_filters * 2**i
        cnn_chain, last_conv = contracting_layer(cnn_chain, neurons)
        contracting_convs.append(last_conv)

    # Middle Layer
    neurons = n_filters * 2**depth
    cnn_chain = middle_layer(cnn_chain, neurons)

    # Expanding Layers
    for i in reversed(range(0, depth)):
        neurons = n_filters * 2**i
        cnn_chain = expanding_layer(cnn_chain, neurons, contracting_convs[i])

    # Output Layer
    conv_out = Conv3D(n_labels, (1, 1, 1), activation=activation)(cnn_chain)
    # Create Model with associated input and output layers
    model = Model(inputs=[inputs], outputs=[conv_out])
    # Return model
    return model

#-----------------------------------------------------#
#                     Subroutines                     #
#-----------------------------------------------------#
# Create a contracting layer
def contracting_layer(input, neurons):
    conv1 = Conv3D(neurons, (3,3,3), activation='relu', padding='same')(input)
    conv2 = Conv3D(neurons, (3,3,3), activation='relu', padding='same')(conv1)
    conc1 = concatenate([input, conv2], axis=4)
    pool = MaxPooling3D(pool_size=(2, 2, 2))(conc1)
    return pool, conv2

# Create the middle layer between the contracting and expanding layers
def middle_layer(input, neurons):
    conv_m1 = Conv3D(neurons, (3, 3, 3), activation='relu', padding='same')(input)
    conv_m2 = Conv3D(neurons, (3, 3, 3), activation='relu', padding='same')(conv_m1)
    conc1 = concatenate([input, conv_m2], axis=4)
    return conc1

# Create an expanding layer
def expanding_layer(input, neurons, concatenate_link):
    up = concatenate([Conv3DTranspose(neurons, (2, 2, 2), strides=(2, 2, 2),
                     padding='same')(input), concatenate_link], axis=4)
    conv1 = Conv3D(neurons, (3, 3, 3), activation='relu', padding='same')(up)
    conv2 = Conv3D(neurons, (3, 3, 3), activation='relu', padding='same')(conv1)
    conc1 = concatenate([up, conv2], axis=4)
    return conc1
