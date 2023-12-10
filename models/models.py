from __future__ import print_function, division, absolute_import
import torch.nn as nn
import torchvision.models as tm
from efficientnet_pytorch import EfficientNet
import timm

# Defining a class named 'Brainmodels' for model management
class Brainmodels:
    def __init__(self, models_name, num_classes, resume_model):
        self.models_name = models_name
        self.num_classes = num_classes
        self.resume_model = resume_model

    #EfficientNet
    def avdprop_net(self):
        if self.resume_model:
            if self.models_name == 'efficientnet-b6':
                model = EfficientNet.from_pretrained(self.models_name, advprop=True, num_classes=self.num_classes)
            else:
                model = EfficientNet.from_pretrained(self.models_name, advprop=False, num_classes=self.num_classes)
            return model
        else:
            if self.models_name == 'efficientnet-b6':
                model = EfficientNet.from_name(self.models_name,  num_classes=self.num_classes)
            else:
                model = EfficientNet.from_name(self.models_name,   num_classes=self.num_classes)
            return model

    # typical model on torchvision
    def tor_models(self):
        model = tm.__dict__[self.models_name](pretrained=self.resume_model)
        return model

    # loading model
    def final_models(self):
        if 'pnasnet5large' in self.models_name:
            model = timm.create_model('pnasnet5large', pretrained=False)


        elif 'efficientnet' in self.models_name:
            model = Brainmodels.avdprop_net(self)

        else:
            model = Brainmodels.tor_models(self)

        return model




