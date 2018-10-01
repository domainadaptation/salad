import torch
import torch.nn as nn

class ConditionalLayer(nn.Module):

    def __init__(self):
        super(ConditionalLayer, self).__init__()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
class ConditionalBatchNorm(ConditionalLayer):
    
    def __init__(self, *args, n_domains = 0, bn_func = nn.BatchNorm2d, **kwargs):
        
        super(ConditionalBatchNorm, self).__init__()
        
        self.n_domains = n_domains
        self.layers    = [bn_func(*args, **kwargs) for i in range(n_domains)]
        
    def _apply(self, fn): 
        super(ConditionalBatchNorm, self)._apply(fn)
        for layer in self.layers:
            layer._apply(fn)
        
    def parameters(self, d=0):
        return self.layers[d].parameters()
        
    def forward(self, x, d):
                
        layer = self.layers[d]
        return layer(x) 

class ConditionalSequential(ConditionalLayer):

    def __init__(self, *modules):

        super(ConditionalSequential, self).__init__()
        self.modulelist   = nn.ModuleList(modules)

    def forward(self, x, *args):

        for module in self.modulelist:
            if isinstance(module, ConditionalLayer):
                x = module(x, *args)
            else:
                x = module(x)
        
        return x

class FeatureAwareNorm2d(ConditionalLayer):
    """ Feature Aware Normalization
    """

    def __init__(self, in_x, in_z, norm_layer="bn"):

        super(FeatureAwareNorm2d, self).__init__()

        # layers
        self.mul_gate = nn.Sequential(
            nn.Conv2d(in_z, in_z, 1),
            nn.ReLU(),
            nn.Conv2d(in_z, in_x, 1),
            #nn.Tanh()
        )
        self.add_gate = nn.Sequential(
            nn.Conv2d(in_z, in_z, 1),
            nn.ReLU(),
            nn.Conv2d(in_z, in_x, 1),
        )

        if norm_layer == "bn":
            self.norm = nn.BatchNorm2d(in_x, momentum=0., affine=False)
        elif norm_layer == "in":
            self.norm = nn.InstanceNorm2d(in_x, momentum=0., affine=False)
        else:
            raise NotImplementedError()

        # parameters
        self.inp_channel = in_x

    def forward(self, x, z = None):
        if z is None:
            x = self.norm(x)
            return x
        
        gamma = self.mul_gate(z)
        beta  = self.add_gate(z)
        x = self.norm(x)
        return torch.mul(x, gamma) + beta

##### ---- Model Definition ---- #####

class FANModel(nn.Module):

    def __init__(self):
        super(FANModel, self).__init__()

        def conv2d_3x3(inp,outp,pad=1):
            return ConditionalSequential(
                nn.Conv2d(inp,outp,kernel_size=3,padding=pad),
                FeatureAwareNorm2d(outp, 128),
                nn.ReLU()
            )

        def conv2d_1x1(inp,outp):
            return ConditionalSequential(
                nn.Conv2d(inp,outp,kernel_size=1,padding=0),
                FeatureAwareNorm2d(outp, 128),
                nn.ReLU()
            )

        def block(inp,outp):
            return ConditionalSequential(
                conv2d_3x3(inp,outp),
                conv2d_3x3(outp,outp),
                conv2d_3x3(outp,outp)
            )

        self.features = ConditionalSequential(
            block(3,128),
            nn.MaxPool2d(2, 2, padding=0),
            nn.Dropout2d(p=0.5),
            block(128,256),
            nn.MaxPool2d(2, 2, padding=0),
            nn.Dropout2d(p=0.5),
            conv2d_3x3(256, 512, pad=0),
            conv2d_1x1(512, 256),
            conv2d_1x1(256, 128),
            nn.AvgPool2d(6, 6, padding=0)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 10)
        )
        
    def forward(self, x):

        phi  = self.features(x, None)
        phi = phi.view(-1,128)
        y = self.classifier(phi)

        return phi, y

class ConditionalModel(nn.Module):

    def __init__(self, n_domains):
        super(ConditionalModel, self).__init__()

        self.n_domains = n_domains

        self.conditional_layers = []

        def bn(n_features):
            layer = ConditionalBatchNorm(n_features, n_domains=self.n_domains)
            self.conditional_layers.append(layer)
            return layer

        def conv2d_3x3(inp,outp,pad=1):
            return ConditionalSequential(
                nn.Conv2d(inp,outp,kernel_size=3,padding=pad),
                bn(outp),
                nn.ReLU()
            )

        def conv2d_1x1(inp,outp):
            return ConditionalSequential(
                nn.Conv2d(inp,outp,kernel_size=1,padding=0),
                bn(outp),
                nn.ReLU()
            )

        def block(inp,outp):
            return ConditionalSequential(
                conv2d_3x3(inp,outp),
                conv2d_3x3(outp,outp),
                conv2d_3x3(outp,outp)
            )

        self.features = ConditionalSequential(
            nn.InstanceNorm2d(3, affine=False,
                momentum=0,
                track_running_stats=False),
            block(3,128),
            nn.MaxPool2d(2, 2, padding=0),
            nn.Dropout2d(p=0.5),
            block(128,256),
            nn.MaxPool2d(2, 2, padding=0),
            nn.Dropout2d(p=0.5),
            conv2d_3x3(256, 512, pad=0),
            conv2d_1x1(512, 256),
            conv2d_1x1(256, 256),
            nn.AvgPool2d(6, 6, padding=0)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 10)
        )
        
    def forward(self, x, d = 0):

        phi  = self.features(x, d)
        # print(phi.size())
        phi = phi.view(-1,256)
        y = self.classifier(phi)

        return phi, y

    def conditional_params(self, d=0):
        for module in self.conditional_layers:
            for p in module.parameters(d):
                yield p

    def parameters(self, d=0, yield_shared=True, yield_conditional=True):
        
        if yield_shared:
            for param in super(ConditionalModel, self).parameters():
                yield param
        
        if yield_conditional:
            for param in self.conditional_params(d):
                yield param
        