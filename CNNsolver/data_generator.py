#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
#External libraries
import keras
#Internal libraries/scripts
import inputreader as CNNsolver_IR

#-----------------------------------------------------#
#              MRI Data Generator (Keras)             #
#-----------------------------------------------------#
# MRI Data Generator for training and predicting (WITH-/OUT segmentation)
## Returns a batch containing multiple patches for each call
class DataGenerator(keras.utils.Sequence):
    # Class Initialization
    def __init__(self, casePointer, data_path, training=False, shuffle=False):
        # Create a working environment from the handed over variables
        self.casePointer = casePointer
        self.data_path = data_path
        self.training = training
        self.shuffle = shuffle
        # Create a counter for MRI internal batch pointer, current Case MRI
        # and global index for the casePointer
        self.idx = -1
        self.batch_pointer = None
        self.current_case = -1
        self.current_mri = None
        # Create a Input Reader instance
        self.reader = CNNsolver_IR.InputReader(data_path)
        # Trigger epoch end once in the beginning
        self.on_epoch_end()

    # Return the next batch for associated index
    def __getitem__(self, params):
        # Increase index
        self.idx += 1
        # Load the next pickled MRI object if necessary
        if self.current_case != self.casePointer[self.idx]:
            self.batch_pointer = 0
            self.current_case = self.casePointer[self.idx]
            self.current_mri = self.reader.case_loader(self.current_case,
                                                       load_seg=self.training,
                                                       pickle=True)
        # Load next volume batch
        batch_vol = self.current_mri.batches_vol[self.batch_pointer]
        # IF batch is for training -> return next vol & seg batch
        if self.training:
            # Load next segmentation batch
            batch_seg = self.current_mri.batches_seg[self.batch_pointer]
            # Update batch_pointer
            self.batch_pointer += 1
            # Return volume and segmentation batch
            return batch_vol, batch_seg
        # IF batch is for predicting -> return next vol batch
        else:
            # Update batch_pointer
            self.batch_pointer += 1
            # Return volume batch
            return batch_vol

    # Return the number of batches for one epoch
    def __len__(self):
        return len(self.casePointer)

    # At every epoch end: Reset batch_pointer and shuffle batches
    def on_epoch_end(self):
        # Reset batch_pointer and index
        self.idx = -1
        self.batch_pointer = 0
        # Shuffle casePointer array
        # if self.shuffle == True:
        #     np.random.shuffle(self.indexes)
