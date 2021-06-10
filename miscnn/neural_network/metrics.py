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
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from tensorflow.keras import backend as K
import numpy as np

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
    return 1-dice_coefficient(y_true, y_pred)

#-----------------------------------------------------#
#                Soft Dice coefficient                #
#-----------------------------------------------------#
def dice_soft(y_true, y_pred, smooth=0.00001):
    # Identify axis
    axis = identify_axis(y_true.get_shape())

    # Calculate required variables
    intersection = y_true * y_pred
    intersection = K.sum(intersection, axis=axis)
    y_true = K.sum(y_true, axis=axis)
    y_pred = K.sum(y_pred, axis=axis)

    # Calculate Soft Dice Similarity Coefficient
    dice = ((2 * intersection) + smooth) / (y_true + y_pred + smooth)

    # Obtain mean of Dice & return result score
    dice = K.mean(dice)
    return dice

def dice_soft_loss(y_true, y_pred):
    return 1-dice_soft(y_true, y_pred)

#-----------------------------------------------------#
#              Weighted Dice coefficient              #
#-----------------------------------------------------#
def dice_weighted(weights):
    weights = K.variable(weights)

    def weighted_loss(y_true, y_pred, smooth=0.00001):
        axis = identify_axis(y_true.get_shape())
        intersection = y_true * y_pred
        intersection = K.sum(intersection, axis=axis)
        y_true = K.sum(y_true, axis=axis)
        y_pred = K.sum(y_pred, axis=axis)
        dice = ((2 * intersection) + smooth) / (y_true + y_pred + smooth)
        dice = dice * weights
        return -dice
    return weighted_loss

#-----------------------------------------------------#
#              Dice & Crossentropy loss               #
#-----------------------------------------------------#
def dice_crossentropy(y_truth, y_pred):
    # Obtain Soft DSC
    dice = dice_soft_loss(y_truth, y_pred)
    # Obtain Crossentropy
    crossentropy = K.categorical_crossentropy(y_truth, y_pred)
    crossentropy = K.mean(crossentropy)
    # Return sum
    return dice + crossentropy

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
    axis = identify_axis(y_true.get_shape())
    tp = K.sum(y_true * y_pred, axis=axis)
    fn = K.sum(y_true * (1-y_pred), axis=axis)
    fp = K.sum((1-y_true) * y_pred, axis=axis)
    tversky_class = (tp + smooth)/(tp + alpha*fn + beta*fp + smooth)
    # Sum up classes to one score
    tversky = K.sum(tversky_class, axis=[-1])
    # Identify number of classes
    n = K.cast(K.shape(y_true)[-1], 'float32')
    # Return Tversky
    return n-tversky

#-----------------------------------------------------#
#             Tversky & Crossentropy loss             #
#-----------------------------------------------------#
def tversky_crossentropy(y_truth, y_pred):
    # Obtain Tversky Loss
    tversky = tversky_loss(y_truth, y_pred)
    # Obtain Crossentropy
    crossentropy = K.categorical_crossentropy(y_truth, y_pred)
    crossentropy = K.mean(crossentropy)
    # Return sum
    return tversky + crossentropy

#-----------------------------------------------------#
#                     Subroutines                     #
#-----------------------------------------------------#
# Identify shape of tensor and return correct axes
def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [1,2,3]
    # Two dimensional
    elif len(shape) == 4 : return [1,2]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


#-----------------------------------------------------#
#                Categorical Focal Loss               #
#-----------------------------------------------------#
def categorical_focal_loss(alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
        Implementation: https://github.com/umbertogriffo/focal-loss-keras/blob/master/src/loss_function/losses.py
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed
