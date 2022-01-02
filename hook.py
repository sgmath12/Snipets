from torch import nn

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        '''
        model : torch model

        ex)
            features = FeatureExtractor(vgg16)
            z = features(x)            # you should forward input x to get features.
            features.activations[0]    # torch tensor [batch size, channel, height, width].      
            features.layer_numbers     # the number of total registered layers.
            features.names[0]          # ex) Conv2d, ReLU, BatchNorm2d.
        '''
        super(FeatureExtractor,self).__init__()
        self.model = model
        self.activations = {} 
        self.names = {}
        self.layer_numbers = 0
        self.register_hook(model)

    def register_hook(self,model):
        idx = 0
        for name,layer in model.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer,nn.ReLU) or isinstance(layer,nn.BatchNorm2d) :
                layer.register_forward_hook(self.get_activation(idx))
                self.names[idx] = layer
                idx += 1
        self.layer_numbers = idx

    def get_activation(self,idx):
        def hook_fn(module,input,output):
            self.activations[idx] = output

        return hook_fn

    def forward(self, x):
        return self.model(x)
        