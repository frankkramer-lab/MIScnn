#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
from keras import backend as K

#-----------------------------------------------------#
#              Standard Dice coefficient              #
#-----------------------------------------------------#
def dice_coefficient(y_true, y_pred, smooth=0.00001):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / \
           (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)

#-----------------------------------------------------#
#             Class-wise Dice coefficient             #
#-----------------------------------------------------#
def dice_classwise(y_true, y_pred, smooth=0.00001):
    intersection = y_true * y_pred
    intersection = K.sum(intersection, axis=[1,2,3])
    y_true = K.sum(y_true, axis=[1,2,3])
    y_pred = K.sum(y_pred, axis=[1,2,3])
    dice = ((2 * intersection) + smooth) / (y_true + y_pred + smooth)
    return dice

def dice_classwise_loss(y_true, y_pred):
    return -dice_classwise(y_true, y_pred)

#-----------------------------------------------------#
#                    Tversky loss                     #
#-----------------------------------------------------#
#                     Reference:                      #
#                Sadegh et al. (2017)                 #
#     Tversky loss function for image segmentation    #
#      using 3D fully convolutional deep networks     #
#-----------------------------------------------------#
# alpha=beta=0.5 : dice coefficient                   #
# alpha=beta=1   : jaccard                            #
# alpha+beta=1   : produces set of F*-scores          #
#-----------------------------------------------------#
def tversky_loss(y_true, y_pred, smooth=0.000001):
    # Define alpha and beta
    alpha = 0.5
    beta  = 0.5
    # Calculate Tversky for each class
    tp = K.sum(y_true * y_pred, axis=[1,2,3])
    fn = K.sum(y_true * (1-y_pred), axis=[1,2,3])
    fp = K.sum((1-y_true) * y_pred, axis=[1,2,3])
    tversky_class = (tp + smooth)/(tp + alpha*fn + beta*fp + smooth)
    # Sum up classes to one score
    tversky = K.sum(tversky_class, axis=[-1])
    # Identify number of classes
    n = K.cast(K.shape(y_true)[-1], 'float32')
    # Return Tversky
    return n-tversky
