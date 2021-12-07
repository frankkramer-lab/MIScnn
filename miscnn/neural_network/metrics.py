#==============================================================================#
#  Author:       Dominik Müller                                                #
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
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np

#-----------------------------------------------------#
#              Standard Dice coefficient              #
#-----------------------------------------------------#
def dice_coefficient(y_true, y_pred, smooth=0.00001):
    """Dice loss originates from Sørensen–Dice coefficient, which is a statistic developed in 1940s to gauge the similarity between two samples.
    Variant: Over all pixels
    Credits documentation: https://github.com/mlyg

    Parameters
    ----------
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """
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
    """Dice loss originates from Sørensen–Dice coefficient, which is a statistic developed in 1940s to gauge the similarity between two samples.
    Variant: Classwise score calculation
    Credits documentation: https://github.com/mlyg

    Parameters
    ----------
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """
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

#-----------------------------------------------------#
#                      Combo Loss                     #
#-----------------------------------------------------#
#                     Reference:                      #
# Michael Yeung, Evis Sala, Carola-Bibiane Schönlieb, #
#               Leonardo Rundo. (2021)                #
#   Unified Focal loss: Generalising Dice and cross   #
#   entropy-based losses to handle class imbalanced   #
#             medical image segmentation              #
#-----------------------------------------------------#
#                Implementation Source:               #
#      https://github.com/mlyg/unified-focal-loss     #
#-----------------------------------------------------#
def combo_loss(alpha=0.5,beta=0.5):
    """Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation
    Link: https://arxiv.org/abs/1805.02798
    Parameters
    ----------
    alpha : float, optional
        controls weighting of dice and cross-entropy loss., by default 0.5
    beta : float, optional
        beta > 0.5 penalises false negatives more than false positives., by default 0.5
    """
    def loss_function(y_true,y_pred):
        dice = dice_coefficient()(y_true, y_pred)
        axis = identify_axis(y_true.get_shape())
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)

        if beta is not None:
            beta_weight = np.array([beta, 1-beta])
            cross_entropy = beta_weight * cross_entropy
        # sum over classes
        cross_entropy = K.mean(K.sum(cross_entropy, axis=[-1]))
        if alpha is not None:
            combo_loss = (alpha * cross_entropy) - ((1 - alpha) * dice)
        else:
            combo_loss = cross_entropy - dice
        return combo_loss

    return loss_function

#-----------------------------------------------------#
#                 Focal Tversky Loss                  #
#-----------------------------------------------------#
#                     Reference:                      #
# Michael Yeung, Evis Sala, Carola-Bibiane Schönlieb, #
#               Leonardo Rundo. (2021)                #
#   Unified Focal loss: Generalising Dice and cross   #
#   entropy-based losses to handle class imbalanced   #
#             medical image segmentation              #
#-----------------------------------------------------#
#                Implementation Source:               #
#      https://github.com/mlyg/unified-focal-loss     #
#-----------------------------------------------------#
def focal_tversky_loss(delta=0.7, gamma=0.75, smooth=0.000001):
    """A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation
    Link: https://arxiv.org/abs/1810.07842
    Parameters
    ----------
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """
    def loss_function(y_true, y_pred):
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1-y_pred), axis=axis)
        fp = K.sum((1-y_true) * y_pred, axis=axis)
        tversky_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
        # Average class scores
        focal_tversky_loss = K.mean(K.pow((1-tversky_class), gamma))

        return focal_tversky_loss

    return loss_function

#-----------------------------------------------------#
#                Symmetric Focal Loss                 #
#-----------------------------------------------------#
#                     Reference:                      #
# Michael Yeung, Evis Sala, Carola-Bibiane Schönlieb, #
#               Leonardo Rundo. (2021)                #
#   Unified Focal loss: Generalising Dice and cross   #
#   entropy-based losses to handle class imbalanced   #
#             medical image segmentation              #
#-----------------------------------------------------#
#                Implementation Source:               #
#      https://github.com/mlyg/unified-focal-loss     #
#-----------------------------------------------------#
def symmetric_focal_loss(delta=0.7, gamma=2.):
    """
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    """
    def loss_function(y_true, y_pred):

        axis = identify_axis(y_true.get_shape())

        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        #calculate losses separately for each class
        back_ce = K.pow(1 - y_pred[:,:,:,0], gamma) * cross_entropy[:,:,:,0]
        back_ce =  (1 - delta) * back_ce

        fore_ce = K.pow(1 - y_pred[:,:,:,1], gamma) * cross_entropy[:,:,:,1]
        fore_ce = delta * fore_ce

        loss = K.mean(K.sum(tf.stack([back_ce, fore_ce],axis=-1),axis=-1))

        return loss

    return loss_function

#-----------------------------------------------------#
#             Symmetric Focal Tversky Loss            #
#-----------------------------------------------------#
#                     Reference:                      #
# Michael Yeung, Evis Sala, Carola-Bibiane Schönlieb, #
#               Leonardo Rundo. (2021)                #
#   Unified Focal loss: Generalising Dice and cross   #
#   entropy-based losses to handle class imbalanced   #
#             medical image segmentation              #
#-----------------------------------------------------#
#                Implementation Source:               #
#      https://github.com/mlyg/unified-focal-loss     #
#-----------------------------------------------------#
def symmetric_focal_tversky_loss(delta=0.7, gamma=0.75):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """
    def loss_function(y_true, y_pred):
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1-y_pred), axis=axis)
        fp = K.sum((1-y_true) * y_pred, axis=axis)
        dice_class = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)

        #calculate losses separately for each class, enhancing both classes
        back_dice = (1-dice_class[:,0]) * K.pow(1-dice_class[:,0], -gamma)
        fore_dice = (1-dice_class[:,1]) * K.pow(1-dice_class[:,1], -gamma)

        # Average class scores
        loss = K.mean(tf.stack([back_dice,fore_dice],axis=-1))
        return loss

    return loss_function

#-----------------------------------------------------#
#                 Asymmetric Focal Loss               #
#-----------------------------------------------------#
#                     Reference:                      #
# Michael Yeung, Evis Sala, Carola-Bibiane Schönlieb, #
#               Leonardo Rundo. (2021)                #
#   Unified Focal loss: Generalising Dice and cross   #
#   entropy-based losses to handle class imbalanced   #
#             medical image segmentation              #
#-----------------------------------------------------#
#                Implementation Source:               #
#      https://github.com/mlyg/unified-focal-loss     #
#-----------------------------------------------------#
def asymmetric_focal_loss(delta=0.7, gamma=2.):
    """For Imbalanced datasets
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    """
    def loss_function(y_true, y_pred):
        axis = identify_axis(y_true.get_shape())

        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)

        #calculate losses separately for each class, only suppressing background class
        back_ce = K.pow(1 - y_pred[:,:,:,0], gamma) * cross_entropy[:,:,:,0]
        back_ce =  (1 - delta) * back_ce

        fore_ce = cross_entropy[:,:,:,1]
        fore_ce = delta * fore_ce

        loss = K.mean(K.sum(tf.stack([back_ce, fore_ce],axis=-1),axis=-1))

        return loss

    return loss_function

#-----------------------------------------------------#
#            Asymmetric Focal Tversky Loss            #
#-----------------------------------------------------#
#                     Reference:                      #
# Michael Yeung, Evis Sala, Carola-Bibiane Schönlieb, #
#               Leonardo Rundo. (2021)                #
#   Unified Focal loss: Generalising Dice and cross   #
#   entropy-based losses to handle class imbalanced   #
#             medical image segmentation              #
#-----------------------------------------------------#
#                Implementation Source:               #
#      https://github.com/mlyg/unified-focal-loss     #
#-----------------------------------------------------#
def asymmetric_focal_tversky_loss(delta=0.7, gamma=0.75):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """
    def loss_function(y_true, y_pred):
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1-y_pred), axis=axis)
        fp = K.sum((1-y_true) * y_pred, axis=axis)
        dice_class = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)

        #calculate losses separately for each class, only enhancing foreground class
        back_dice = (1-dice_class[:,0])
        fore_dice = (1-dice_class[:,1]) * K.pow(1-dice_class[:,1], -gamma)

        # Average class scores
        loss = K.mean(tf.stack([back_dice,fore_dice],axis=-1))
        return loss

    return loss_function

#-----------------------------------------------------#
#             Symmetric Unified Focal Loss            #
#-----------------------------------------------------#
#                     Reference:                      #
# Michael Yeung, Evis Sala, Carola-Bibiane Schönlieb, #
#               Leonardo Rundo. (2021)                #
#   Unified Focal loss: Generalising Dice and cross   #
#   entropy-based losses to handle class imbalanced   #
#             medical image segmentation              #
#-----------------------------------------------------#
#                Implementation Source:               #
#      https://github.com/mlyg/unified-focal-loss     #
#-----------------------------------------------------#
def sym_unified_focal_loss(weight=0.5, delta=0.6, gamma=0.5):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to symmetric Focal Tversky loss and symmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    """
    def loss_function(y_true,y_pred):
      symmetric_ftl = symmetric_focal_tversky_loss(delta=delta, gamma=gamma)(y_true,y_pred)
      symmetric_fl = symmetric_focal_loss(delta=delta, gamma=gamma)(y_true,y_pred)
      if weight is not None:
        return (weight * symmetric_ftl) + ((1-weight) * symmetric_fl)
      else:
        return symmetric_ftl + symmetric_fl

    return loss_function

#-----------------------------------------------------#
#            Asymmetric Unified Focal Loss            #
#-----------------------------------------------------#
#                     Reference:                      #
# Michael Yeung, Evis Sala, Carola-Bibiane Schönlieb, #
#               Leonardo Rundo. (2021)                #
#   Unified Focal loss: Generalising Dice and cross   #
#   entropy-based losses to handle class imbalanced   #
#             medical image segmentation              #
#-----------------------------------------------------#
#                Implementation Source:               #
#      https://github.com/mlyg/unified-focal-loss     #
#-----------------------------------------------------#
def asym_unified_focal_loss(weight=0.5, delta=0.6, gamma=0.5):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to asymmetric Focal Tversky loss and asymmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    """
    def loss_function(y_true,y_pred):
      asymmetric_ftl = asymmetric_focal_tversky_loss(delta=delta, gamma=gamma)(y_true,y_pred)
      asymmetric_fl = asymmetric_focal_loss(delta=delta, gamma=gamma)(y_true,y_pred)
      if weight is not None:
        return (weight * asymmetric_ftl) + ((1-weight) * asymmetric_fl)
      else:
        return asymmetric_ftl + asymmetric_fl

    return loss_function

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
