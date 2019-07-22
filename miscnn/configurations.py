#-----------------------------------------------------#
#               Configurations - Object               #
#-----------------------------------------------------#
#class Configurations:
config = dict()
# Dataset
config["data_path"] = "data"                    # Path to the input data dir
config["model_path"] = "model"                  # Path to the model data dir
config["output_path"] = "predictions"           # Path to the predictions directory
config["evaluation_path"] = "evaluation"        # Path to the evaluation directory
# GPU Architecture
config["gpu_number"] = 1                        # Number of GPUs (if > 2 = multi GPU)
# Neural Network Settings
config["dimension"] = "3D"                      # Model Dimension (2D or 3D)
config["architecture"] = "unet"                 # Neural Network architecture
config["model_variant"] = "standard"            # Model variant of the architecture
config["input_shape"] = (None, 16, 16, 1)       # Neural Network input shape
config["patch_size"] = (16, 16, 16)             # Patch shape/size
config["classes"] = 3                           # Number of output classes
config["batch_size"] = 16                       # Number of patches in on step
# Training
config["epochs"] = 5                            # Number of epochs for training
config["max_queue_size"] = 3                    # Number of preprocessed batches
config["learninig_rate"] = 0.0001               # Learninig rate for the training
config["shuffle"] = True                        # Shuffle batches for training
# Data Augmentation
config["overlap"] = (0,0,0)                     # Overlap in (x,y,z)-axis
config["skip_blanks"] = True                    # Skip patches with only background
config["scale_input_values"] = False            # Scale volume values to [0,1]
config["rotation"] = False                      # Rotate patches in 90/180/270°
config["flipping"] = False                      # Reflect/Flip patches
config["flip_axis"] = (3)                       # Define the flipping axes (x,y,z <-> 1,2,3)
# Prediction
config["pred_overlap"] = False                  # Usage of overlapping patches in prediction
# Evaluation
config["n_folds"] = 3                           # Number of folds for cross-validation
config["per_split"] = 0.20                      # Percentage of Testing Set for split-validation
config["n_loo"] = 1                             # Number of cycles for leave-one-out
config["visualize"] = True                      # Print out slice images for visual evaluation
config["class_freq"] = False                    # Calculate the class frequencies for each slice

#-----------------------------------------------------#
#              Configurations - Function              #
#-----------------------------------------------------#
def get_options():
    return config
