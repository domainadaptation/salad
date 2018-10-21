# How to contribute

There are various ways in which you can contribute to `salad`, depending on your background.
In general, we invite all researchers and software engineerings interested in transfer learning, semi-supervised learning and domain adaptation to contribute to `salad`.

There are several ways in which contribution is possible.

### Opening an Issue

A great way of helping up improving `salad` is by proposing new directions for extending the library that might benefit your usecases the most, or reflect the research results you got.
Here are some examples:

- You are a machine learning researcher and miss an important algorithms? Please open an [issue](https://github.com/domainadaptation/salad/issues) and tell us more about the algorithm, including a link to the original paper and preferably a reference implementation, if available.
- You recently published a paper on transfer learning or the related fields? Please reach out to us and post a new [issue](https://github.com/domainadaptation/salad/issues) with the `enhancement` tag. You might also consider to add your paper to the [reading list](https://github.com/domainadaptation/awesome-transfer-learning).
- You are author of one of the already integrated algorithms and think our evaluation scheme is unfair/does not match the results in your paper? While it is true that the benchmarking results currently up on domainadaptation.org do not always match the exact numbers reported in the paper, this might be due to the use of different architectures or different hyperparameters. In general, we tried to ensure that the implementations in `salad` are capable of reproducing reported numbers. For a configuration were this is possible (using possibly a model architecture or training scheme not yet supported by `salad`), have a look at the `examples` folder. If you still have the impression that something might we wrong with the implementation, please open an [issue](https://github.com/domainadaptation/salad/issues).
- Of course, if you want to test our library and experience any malfunction or bugs, please open an [issue](https://github.com/domainadaptation/salad/issues) as well under the `bug` tag.

### Software Contributions

If you want to dedicate more time and help us extending `salad`, please feel free to send us pull requests with new algorithms. Prior to working on a new feature, please open an issue, and mention that you are working on this particular new feature.
This helps us coordinating efforts to extend the library.

Specifically, we invite contributions for

- translation based domain adaptation algorithms
- new dataset loaders, such as Cityscapes or the Amazon Office-31 dataset, which is commonly used for benchmarking
- helper functions for transfer learning
- application areas beyond computer vision

## Contact

`salad` is currently maintained by [Steffen Schneider](http://stes.io).
If you have any questions not covered in this contribution guide, please contact me directly.
