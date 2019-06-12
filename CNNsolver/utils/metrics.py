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
# Ref: salehi17, "Tversky loss function for image segmentation using 3D FCDN"
# -> the score is computed for each class separately and then summed
# alpha=beta=0.5 : dice coefficient
# alpha=beta=1   : tanimoto coefficient (also known as jaccard)
# alpha+beta=1   : produces set of F*-scores
# implemented by E. Moebel, 06/04/18
def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta  = 0.5

    ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true

    num = K.sum(p0*g0, (0,1,2,3))
    den = num + alpha*K.sum(p0*g1,(0,1,2,3)) + beta*K.sum(p1*g0,(0,1,2,3))

    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T
