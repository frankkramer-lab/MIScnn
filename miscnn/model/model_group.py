from miscnn.model.model import Model

#-----------------------------------------------------#
#            Neural Network (model) class             #
#-----------------------------------------------------#
# Class which represents the Neural Network and which run the whole pipeline
class Model_Group(Model):
    
    __CNTR = 0
    
    def __init__ (self, models, preprocessor, verify_preprocessor=True):
        Model.__init__(self, preprocessor)
        
        self.models = models
        
        self.id = Model_Group.__CNTR++ #technically not thread safe
        
        if (verify_preprocessor):
            for model in self.models:
                if not model.preprocessor == self.preprocessor:
                    raise RuntimeError("not all models use the same preprocessor. This can have have unintended effects and instabilities. To disable this warning pass \"verify_preprocessor=False\"")
        
    
    def train(self, sample_list, epochs=20, iterations=None, callbacks=[]):
        raise NotImplementedError()
    
    def predict(self, sample_list, activation_output=False):
        raise NotImplementedError()
    # Evaluate the Model using the MIScnn pipeline
    def evaluate(self, training_samples, validation_samples, evaluation_path="evaluation", epochs=20, iterations=None, callbacks=[], store=True):
        subdir = create_directories(evaluation_path, "model_" + str(self.id))
        
        for model in self.models:
            model.evaluate(training_samples, validation_samples, evaluation_path=subdir, epochs=epochs, iterations=iterations, callbacks=callbacks)
    # Dump model to file
    def dump(self, file_path):
        raise NotImplementedError()

    # Load model from file
    def load(self, file_path, custom_objects={}):
        raise NotImplementedError()