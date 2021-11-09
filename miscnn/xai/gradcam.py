from miscnn.data_loading.sample import Sample
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from miscnn.utils.visualizer import visualize_samples
from miscnn.utils.patch_operations import concat_matrices

def generateGradientMap(preprocessed_input, model, cls = 1, three_dim=True, abs_w = False, posit_w = False, normalize = False):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.model.inputs], [model.model.layers[-2].output, model.model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(preprocessed_input)
        A = preds[..., cls]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
    grads = tape.gradient(A, last_conv_layer_output)
    
    A, grads = A[0, :], grads[0]

    """Defines a matrix of alpha^k_c. Each alpha^k_c denotes importance (weights) of a feature map A^k for class c.
    If abs_w=True, absolute values of the matrix are processed and returned as weights.
    If posit_w=True, ReLU is applied to the matrix."""
    if (three_dim):
        alpha_c = np.mean(grads[0, ...], axis=(0, 1, 2))
    else:
        alpha_c = np.mean(grads[0, ...], axis=(0, 1))
    if abs_w:
        alpha_c = abs(alpha_c)
    if posit_w:
        alpha_c = np.maximum(alpha_c, 0)

    """The last step to get the activation map. Should be called after outputGradients and gradientWeights."""
    # weighted sum of feature maps: sum of alpha^k_c * A^k
    cam = np.dot(A, alpha_c)  # *abs(grads) or max(grads,0)
    # apply ReLU to te sum
    cam = np.maximum(cam, 0)
    # normalize non-negative weighted sum
    cam_max = cam.max()
    if cam_max != 0 and normalize:
        cam = cam / cam_max
    
    return np.expand_dims(cam, -1)


def visualizeGradientHeatmap(sample_list, model, cls = 1, three_dim=True, abs_w = False, posit_w = False, normalize = False, out_dir = "vis", progress = False):
    pp = model.preprocessor
    skip_blanks = pp.patchwise_skip_blanks
    pp.patchwise_skip_blanks = False
    preprocessed_input = [pp.run([s], training=False) for s in sample_list]
    pp.patchwise_skip_blanks = skip_blanks
    preprocessed_input = [[a[0] for a in s] for s in preprocessed_input]
    t = [len(s) + 2 for s in preprocessed_input]
    patches = sum(t)
    pbar = tqdm(total=patches)
    
    for index, sample in zip(sample_list, preprocessed_input):
        p = []
        
        for patch in sample:
            patchPred = generateGradientMap(patch, model, cls, three_dim, abs_w, posit_w, normalize)
            p.append(patchPred)
            pbar.update(1)
        
        npp = np.asarray(p)
        
        npp -= npp.min() #account for negative weights
        
        s = pp.data_io.sample_loader(index, False, False)
        s.index = s.index + ".gradcam"
        s.pred_data = pp.postprocessing(pp.cache.pop(index), npp, activation_output=True)
        pbar.update(1)
        visualize_samples([s], mask_pred=True)
        pbar.update(1)
