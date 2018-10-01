from torchvision.datasets import SVHN as SVHNBase

class SVHN(SVHNBase):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    @property
    def images(self):
        return self.data