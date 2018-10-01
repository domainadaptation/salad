#/bin/bash

script=ensembling
path=/gpfs01/bethge/home/sschneider/thesis/code/domainadaptation/projects/Papers

ssh gpu16 docker exec sschneider1 "python3 $path/$script/train_${script}.py --log $path/$script/log"