import miscnn.utils.visualizer as vis
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.animation as animation
import functools

#compute gradient weighted distance to class

def is_activation_pred(pred_data, expected_classes):
    if pred_data is None:
        return False
    if not pred_data.shape[-1] == expected_classes:
        print("Expected " + str(expected_classes) + " classes but found activation of shape " + str(pred_data.shape))
        return False
    if not np.issubdtype(pred_data.dtype, np.floating):
        print("Activation Array is not using a floating-point value type. Assuming this is erroneous data.")
        return False
    return True
        

def display_2D(out_path, fig, axes, grad_overlay, activation_overlay = None):
    grad_overlay = grad_overlay.astype(np.uint8)
    if activation_overlay is not None:
        activation_overlay = activation_overlay.astype(np.uint8)
    
    fig.supxlabel('Classes')
    
    if activation_overlay is not None:
        imgs = [[img.imshow(activation_overlay[cls, i]) for cls, img in enumerate(img)] if k == 0 else [img.imshow(grad_overlay[cls, i]) for cls, img in enumerate(img_1)] for k, img_1 in enumerate(axes)]
    else:
        imgs = [img.imshow(grad_overlay[cls, i]) for cls, img in enumerate(axes)]
    

def display_3D(out_path, fig, axes, grad_overlay, activation_overlay = None):
    grad_overlay = grad_overlay.astype(np.uint8)
    if activation_overlay is not None:
        activation_overlay = activation_overlay.astype(np.uint8)
    
    fig.supxlabel('Classes')
    
    data = np.zeros(grad_overlay.shape[2:4])
    
    
    if activation_overlay is not None:
        fig.supylabel('Gradient vs Activation')
        imgs = [[a.imshow(data) for a in _a] for _a in axes] #axes are assumed to be 2D
        # Update function for both images to show the slice for the current frame
        def update(imgs, i):
            for _a in axes:
                for a in _a:
                    a.set_title("Slice: " + str(i))
            imgs = [[im.set_data(activation_overlay[cls, i]) for cls, im in enumerate(img_1)] if k == 0 else [im.set_data(grad_overlay[cls, i]) for cls, im in enumerate(img_1)] for k, img_1 in enumerate(imgs)]
            return imgs
    else:
        fig.supylabel('Gradients')
        imgs = [a.imshow(data) for a in axes] #axes are assumed to be flat
        # Update function for both images to show the slice for the current frame
        def update(imgs, i):
            for a in axes:
                a.set_title("Slice: " + str(i))
            imgs = [img.set_data(grad_overlay[cls, i]) for cls, img in enumerate(imgs)]
            return imgs
    # Compute the animation (gif)
    ani = animation.FuncAnimation(fig, functools.partial(update, imgs), frames=len(grad_overlay[0]), interval=10,
                                  repeat_delay=0, blit=False)
    
    # Save the animation (gif)
    ani.save(out_path, writer='imagemagick', fps=30)


def compute_certainty_score(*args):
    s = 0
    
    m = 1 - np.max(args)
    
    for i in args:
        n = i + m
        s += n * n 
    return np.sqrt(s)
    
vectorized_compute_certainty_score = np.vectorize(compute_certainty_score)


def group_visualization(sample, gradcam, three_dim=True, out_dir = "vis", method="grid_display", alpha = 0.3):
    
    display = display_2D
    if three_dim:
        display = display_3D
    
    sample = vis.to_samples([sample])[0] #normalize data to sample object
    
    gradcam = np.array(gradcam) #this is done to ensure that the operations are not run on a view but on data
    
    if not type(gradcam) == np.ndarray:
        raise ValueError("expected a collection of Gradcam visualizations or predictions")
    
    expected_classes = gradcam.shape[0]
    
    if method == "grid_display": #display all images that are generated as a Grid.
        
        grad_overlayed = np.zeros(gradcam.shape + (3,))
        
        for i in range(gradcam.shape[0]):
            grad_overlayed[i] = vis.overlay_segmentation_greyscale(sample.img_data, gradcam[i], alpha = alpha)
        
        if not is_activation_pred(sample.pred_data, expected_classes):
            print("prediction data not found. Visualizing without.")
            
            fig, axes = plt.subplots(1, expected_classes)
            
            # Set up the output path for the gif
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            file_name = "visualization.comp.case_" + str(sample.index).zfill(5) + ".gif"
            out_path = os.path.join(out_dir, file_name)
            
            display(out_path, fig, axes, grad_overlayed)
            
        else:
            activation_overlayed = np.zeros((sample.pred_data.shape[-1],) + sample.pred_data.shape[:-1] + (3,))
            
            for i in range(sample.pred_data.shape[-1]):
                activation_overlayed[i] = vis.overlay_segmentation_greyscale(sample.img_data, sample.pred_data[..., i], alpha = alpha)
        
            
            fig, axes = plt.subplots(2, expected_classes)
            
            # Set up the output path for the gif
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            file_name = "visualization.grad.comp.case_" + str(sample.index).zfill(5) + ".gif"
            out_path = os.path.join(out_dir, file_name)
            
            display(out_path, fig, axes, grad_overlayed, activation_overlayed)
            
    elif method == "grid_certainty":
        if not is_activation_pred(sample.pred_data, expected_classes):
            raise ValueError("Activation Output of the neural network prediction is required in the sample.")
        
        #the certainty map distance is disported from a (hyper-)sphere to a (hyper-)cube. the intention is to weigh uncertain areas higher
        certainty_map = vectorized_compute_certainty_score(*[sample.pred_data[..., i] for i in range(sample.pred_data.shape[-1])])
        
        normalized_act = sample.pred_data / np.expand_dims(np.linalg.norm(sample.pred_data, axis = -1), -1)
        normalized_grad = gradcam / np.expand_dims(np.linalg.norm(gradcam, axis = 0) + 0.1, 0) #note the epsilon to avoid division by zero
        
        normalized_act = np.moveaxis(normalized_act, -1, 0)
        
        angle = np.sum(np.multiply(normalized_act, normalized_grad), axis = 0) #compute dot product along the vectors of the according dimension
        
        #angle = np.multiply(np.expand_dims(angle, -1), np.expand_dims(certainty_map, -1))
        
        certainty_map = vis.normalize(certainty_map)
        angle = vis.normalize(angle)
        
        angle = np.stack([angle, certainty_map], axis = 0)
        
        angle = np.stack([angle, angle, angle], axis = -1)
        
        fig, axes = plt.subplots(1, 2)
        
        # Set up the output path for the gif
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        file_name = "visualization.certainty.case_" + str(sample.index).zfill(5) + ".gif"
        out_path = os.path.join(out_dir, file_name)
        
        display(out_path, fig, axes, angle)
        
        pass #compute certainty score of activation map. use certainty to weight activation. then compute angle between activation and gradient vector
    elif method == "col_mapping":
        
        if not gradcam.shape[0] == 3:
            raise ValueError("Currently anything but a 1:1 mappping is not supported")
        
        
        gradcam = np.moveaxis(gradcam, 0, -1)
        
        vol_rgb = np.stack([sample.img_data, sample.img_data, sample.img_data], axis=-1)
        segbin = gradcam >= 0
        
        # Weighted sum where there's a value to overlay
        grad_overlayed = np.where(
            segbin,
            np.round(alpha*gradcam+(1-alpha)*vol_rgb).astype(np.uint8),
            np.round(vol_rgb).astype(np.uint8)
        )
        
        grad_overlayed = np.expand_dims(grad_overlayed, 0)
        
        if not is_activation_pred(sample.pred_data, expected_classes):
            print("prediction data not found. Visualizing without.")
            fig, axes = plt.subplots(1, 1)
            
            # Set up the output path for the gif
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            file_name = "visualization.grad.col_map.case_" + str(sample.index).zfill(5) + ".gif"
            out_path = os.path.join(out_dir, file_name)
            
            display(out_path, fig, [axes], grad_overlayed)
        else:
            pred = sample.pred_data
            
            segbin = pred >= 0
            
            # Weighted sum where there's a value to overlay
            act_overlayed = np.where(
                segbin,
                np.round(alpha*pred+(1-alpha)*vol_rgb).astype(np.uint8),
                np.round(vol_rgb).astype(np.uint8)
            )
            act_overlayed = np.expand_dims(act_overlayed, 0)
            fig, axes = plt.subplots(2, 1)
            
            # Set up the output path for the gif
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            file_name = "visualization.grad.col_map.case_" + str(sample.index).zfill(5) + ".gif"
            out_path = os.path.join(out_dir, file_name)
            
            display(out_path, fig, [[a] for a in axes], grad_overlayed, act_overlayed)
        
        pass #map each class to a color channel for both activation and gradient
    elif method == "col_map_activ_delta":
        if not is_activation_pred(sample.pred_data):
            raise ValueError("Activation Output of the neural network prediction is required in the sample.")
        pass #split classes into color channels, then compute intensity using activation/gradient delta
    elif method == "off_class_grads":
        if not is_activation_pred(sample.pred_data, expected_classes):
            raise ValueError("Activation Output of the neural network prediction is required in the sample.")
        pass #display gradients mean/maximum/sum of gradients for each pixel 
        
        
