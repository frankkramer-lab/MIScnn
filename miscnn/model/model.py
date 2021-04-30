

class Model():
    
    
    
    def __init__ (self, preprocessor):
        self.preprocessor = preprocessor
    
    
    
    def train(self, sample_list, epochs=20, iterations=None, callbacks=[]):
        raise NotImplementedError()
    
    def predict(self, sample_list, activation_output=False):
        raise NotImplementedError()
    # Evaluate the Model using the MIScnn pipeline
    def evaluate(self, training_samples, validation_samples, evaluation_path="evaluation", epochs=20, iterations=None, callbacks=[], store=True):
        raise NotImplementedError()
    # Dump model to file
    def dump(self, file_path):
        raise NotImplementedError()

    # Load model from file
    def load(self, file_path, custom_objects={}):
        raise NotImplementedError()
    