from miscnn.model.model import Model

#-----------------------------------------------------#
#            Neural Network (model) class             #
#-----------------------------------------------------#
# Class which represents the Neural Network and which run the whole pipeline
class Model_Group(Model):
    
    def __init__ (self, preprocessor):
        Model.__init__(self, preprocessor)
    