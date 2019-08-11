#-----------------------------------------------------#
#                     Reference:                      #
#   Olaf Ronneberger, Philipp Fischer, Thomas Brox.   #
#                    18 May 2015.                     #
#          U-Net: Convolutional Networks for          #
#            Biomedical Image Segmentation.           #
#                    MICCAI 2015.                     #
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
    pool = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    return pool, conv2

# Create the middle layer between the contracting and expanding layers
def middle_layer(input, neurons):
    conv_m1 = Conv3D(neurons, (3, 3, 3), activation='relu', padding='same')(input)
    conv_m2 = Conv3D(neurons, (3, 3, 3), activation='relu', padding='same')(conv_m1)
    return conv_m2

# Create an expanding layer
def expanding_layer(input, neurons, concatenate_link):
    up = concatenate([Conv3DTranspose(neurons, (2, 2, 2), strides=(2, 2, 2),
                     padding='same')(input), concatenate_link], axis=4)
    conv1 = Conv3D(neurons, (3, 3, 3), activation='relu', padding='same')(up)
    conv2 = Conv3D(neurons, (3, 3, 3), activation='relu', padding='same')(conv1)
    return conv2
