🥗 salad 
========

**S**\ emi-supervised **A**\ daptive **L**\ earning **A**\ cross **D**\ omains

.. figure:: docs/img/domainshift.png
   :alt: 


``salad`` is a library to easily setup experiments using the current
state-of-the art techniques in domain adaptation. It features several of
recent approaches, with the goal of being able to run fair comparisons
between algorithms and transfer them to real-world use cases. The
toolbox is under active development and will extended when new
approaches are published.

Contribute on Github: `https://github.com/domainadaptation/salad`_

Currently implements the following techniques (in ``salad.solver``)

-  VADA (``VADASolver``),
   `arxiv:1802.08735 <https://arxiv.org/abs/1802.08735>`__
-  DIRT-T (``DIRTTSolver``),
   `arxiv:1802.08735 <https://arxiv.org/abs/1802.08735>`__
-  Self-Ensembling for Visual Domain Adaptation
   (``SelfEnsemblingSolver``)
   `arxiv:1706.05208 <https://arxiv.org/abs/1706.05208>`__
-  Associative Domain Adaptation (``AssociativeSolver``),
   `arxiv:1708.00938 <https://arxiv.org/pdf/1708.00938.pdf>`__
-  Domain Adversarial Training (``DANNSolver``),
   `jmlr:v17/15-239.html <http://jmlr.org/papers/v17/15-239.html>`__
-  Generalizing Across Domains via Cross-Gradient Training
   (``CrossGradSolver``),
   `arxiv:1708.00938 <http://arxiv.org/abs/1804.10745>`__
-  Adversarial Dropout Regularization (``AdversarialDropoutSolver``),
   `arxiv.org:1711.01575 <https://arxiv.org/abs/1711.01575>`__

Implements the following features (in ``salad.layers``):

-  Weights Ensembling using Exponential Moving Averages or Stored
   Weights
-  WalkerLoss and Visit Loss
   (`arxiv:1708.00938 <https://arxiv.org/pdf/1708.00938.pdf>`__)
-  Virtual Adversarial Training
   (`arxiv:1704.03976 <https://arxiv.org/abs/1704.03976>`__)

Coming soon:

-  Deep Joint Optimal Transport (``DJDOTSolver``),
   `arxiv:1803.10081 <https://arxiv.org/abs/1803.10081>`__
-  Translation based approaches

💻 Installation
---------------

Requirements can be found in ``requirement.txt`` and can be installed
via

.. code:: bash

    pip install -r requirements.txt

Install the package via

.. code:: bash

    pip install torch-salad

For the latest development version, install via

.. code:: bash

    pip install git+https://github.com/bethgelab/domainadaptation


📚 Using this library
---------------------

Along with the implementation of domain adaptation routines, this
library comprises code to easily set up deep learning experiments in
general. 

This section will be extended upon pre-release.

Quick Start
~~~~~~~~~~~

To get started, the ``scripts/`` directory contains several python scripts
for both running replication studies on digit benchmarks and studies on
a different dataset (toy example: adaptation to noisy images).

.. code:: bash

    $ cd scripts
    $ python train_digits.py
    $ python train_noise.py


Reasons for using solver abstractions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The chosen abstraction style organizes experiments into a subclass of
``Solver``.

Quickstart: MNIST Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As a quick MNIST experiment:

.. code:: python

    from salad.solvers import Solver

    class MNISTSolver(Solver):

        def __init__(self, model, dataset, **kwargs):

            self.model = model
            super().__init__(dataset, **kwargs)

        def _init_optims(self, lr = 1e-4, **kwargs):
            super()._init_optims(**kwargs)

            opt = torch.optim.Adam(self.model.parameters(), lr = lr)
            self.register_optimizer(opt)

        def _init_losses(self):
            pass

For a simple tasks as MNIST, the code is quite long compared to other
PyTorch examples `TODO <#>`__.

💡 Domain Adaptation Problems
-----------------------------

Legend: Implemented (✓), Under Construction (🚧)

📷 Vision
~~~~~~~~~

-  Digits: MNIST ↔ SVHN ↔ USPS ↔ SYNTH (✓)
-  `VisDA 2018 Openset and Detection <http://ai.bu.edu/visda-2018>`__
   (✓)
-  Synthetic (GAN) ↔ Real (🚧)
-  CIFAR ↔ STL (🚧)
-  ImageNet to
   `iCubWorld <https://robotology.github.io/iCubWorld/#datasets>`__ (🚧)

🎤 Audio
~~~~~~~~

-  `Mozilla Common Voice Dataset <https://voice.mozilla.org/>`__ (🚧)

፨ Neuroscience
~~~~~~~~~~~~~~

-  White Noise ↔ Gratings ↔ Natural Images (🚧)
-  `Deep Lab Cut Tracking <https://github.com/AlexEMG/DeepLabCut>`__ (🚧)

🔗 References to open source software
-------------------------------------

Part of the code in this repository is inspired or borrowed from
original implementations, especially:

-  https://github.com/Britefury/self-ensemble-visual-domain-adapt
-  https://github.com/Britefury/self-ensemble-visual-domain-adapt-photo/
-  https://github.com/RuiShu/dirt-t
-  https://github.com/gpascualg/CrossGrad
-  https://github.com/stes/torch-associative
-  https://github.com/haeusser/learning\_by\_association
-  https://mil-tokyo.github.io/adr\_da/

Excellent list of domain adaptation ressources: -
https://github.com/artix41/awesome-transfer-learning

👤 Contact
----------

Maintained by `Steffen Schneider <https://code.stes.io>`__. Work is part
of my thesis project at the `Bethge Lab <http://bethgelab.org>`__. This
README is also available as a webpage at
`salad.domainadaptation.org <http://salad.domainadaptation.org>`__. We
welcome issues and pull requests `to the official github
repository <https://github.com/bethgelab/domainadaptation>`__.
