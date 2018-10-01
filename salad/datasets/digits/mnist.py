from torchvision.datasets import MNIST as MNISTBase


class MNIST(MNISTBase):
    """ MNIST Dataset
    """

    def __init__(self, root, split = 'train', transform = None, label_transform = None, download=True):

        super().__init__(root=root, train = (split == 'train'),
                         transform = transform,
                         download=download)

    @property
    def images(self):
        if self.train:
            return self.train_data
        else:
            return self.test_data

    @property
    def labels(self):
        if self.train:
            return self.train_labels
        else:
            return self.test_labels
