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
def Unet(input_shape, n_labels, activation='sigmoid'):
    inputs = Input(input_shape)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    conc1 = concatenate([inputs, conv1], axis=4)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conc1)

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    conc2 = concatenate([pool1, conv2], axis=4)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conc2)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    conc3 = concatenate([pool2, conv3], axis=4)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conc3)

    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
    conc4 = concatenate([pool3, conv4], axis=4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conc4)

    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)
    conc5 = concatenate([pool4, conv5], axis=4)

    up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2),
                      padding='same')(conc5), conv4], axis=4)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)
    conc6 = concatenate([up6, conv6], axis=4)

    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2),
                      padding='same')(conc6), conv3], axis=4)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)
    conc7 = concatenate([up7, conv7], axis=4)

    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2),
                      padding='same')(conc7), conv2], axis=4)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv8)
    conc8 = concatenate([up8, conv8], axis=4)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2),
                      padding='same')(conc8), conv1], axis=4)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv9)
    conc9 = concatenate([up9, conv9], axis=4)

    conv10 = Conv3D(n_labels, (1, 1, 1), activation=activation)(conc9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model
