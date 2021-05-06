

class Model():
    __CNTR = 0
    
    def __init__ (self, preprocessor):
        self.preprocessor = preprocessor
        # Identify data parameters
        self.three_dim = preprocessor.data_io.interface.three_dim
        self.channels = preprocessor.data_io.interface.channels
        self.classes = preprocessor.data_io.interface.classes
        
        self.id = Model.__CNTR
        Model.__CNTR += 1 #technically not thread safe
        
    
    
    
    def train(self, sample_list, epochs=20, iterations=None, callbacks=[]):
        raise NotImplementedError()
    
    def predict(self, sample_list, activation_output=False):
        raise NotImplementedError()
    # Evaluate the Model using the MIScnn pipeline
    def evaluate(self, training_samples, validation_samples, evaluation_path="evaluation", epochs=20, iterations=None, callbacks=[], store=True):
        raise NotImplementedError()
    
    def reset(self):
        raise NotImplementedError()
    
    def copy(self):
        raise NotImplementedError()
    
    # Dump model to file
    def dump(self, file_path):
        raise NotImplementedError()
    # Load model from file
    def load(self, file_path, custom_objects={}):
        raise NotImplementedError()
    