from torchvision import transforms, datasets

import sys
#sys.path.append('/gpfs01/bethge/home/sschneider/thesis/code/domainadaptation/')
sys.path.append('/home/stes/code/thesis/code/domainadaptation/')

from torchvision import transforms
import torch
import salad.datasets

from salad.solver.da import CrossGradSolver, Model
from salad.utils import config

def get_data(batch_size = 64, shuffle = True, num_workers = 0, train=True, test_angle = 0):

    data = []

    angles = list(range(0,90,15))
    
    print("testing on {}".format(angles[test_angle]))
    if train:
        del(angles[test_angle])
        print('training on {}'.format(",".join(str(i) for i in angles)))
    else:
        angles = [angles[test_angle]]
        print('testing on {}'.format(",".join(str(i) for i in angles)))   
    
    noisemodels = [
        transforms.RandomRotation([i-1,i+1]) for i in angles
    ]

    for N in noisemodels:

        transform = transforms.Compose([
            N,    
            transforms.ToTensor(),  
                transforms.Normalize(mean=(0.43768448, 0.4437684,  0.4728041 ),
                                    std= (0.19803017, 0.20101567, 0.19703583))
        ])
        mnist = datasets.MNIST('./data', train=train, download=True, transform=transform)

        data.append(torch.utils.data.DataLoader(
            mnist, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers))
    
    loader = salad.datasets.JointLoader(*data, collate_fn = collate)
    
    return loader

class CrossGradConfig(config.DomainAdaptConfig):
    
    def _init(self):
        self.add_argument('--testangle', default=0, type=int, help="Test Angle")

if __name__ == '__main__':
    args = CrossGradConfig("Cross Grad Solver")

    for angle in range(6):
        model = Model(10, 5)
        data  = get_data(test_angle = angle, num_workers = 4)
        solver = CrossGradSolver(model, data, gpu=0, n_epochs = 10, savedir = args.log + '-{}'.format(angle))
        solver.optimize()