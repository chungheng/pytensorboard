# Python wrapper for Tensorflow Summary object

This package aims to leverage the visualization tool
[Tensorboard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard)
for monitoring iterative procedures, such as update step in machine learning
algorithm (implemented in packages other than Tensorflow,
ex. [Theano](https://github.com/Theano/Theano)
or [PyTorch](https://pytorch.org/)).
The package provides python APIs to log data as Tensorflow
[tf.Summary](https://www.tensorflow.org/api_docs/python/tf/summary) objects into `Tensorflow` event files. The event files can be visualized in `Tensorboard`.

Note this package **DOES NOT** support visualization of the network graph.
## Installation

Clone and navigate into the repository, and then execute the command:

    python setup.py

This package depends on `Tensorflow` and `TensorBoard`. Installing `Tensorflow` through `pip` should include `TensorBoard`.

## Usage

Logging and visualizing data with this package takes three steps:

1. Create an `TFSummary` object with a given directory.
2. Log data with the `TFSummary` object in iterative steps.
3. Launch a `TensorBoard` server with the same directory in Step 1.

### Instantiating TFSummary

The `TFSummary` object is designed to be similar to the Tensorflow `FileWriter` object. The constructor of `TFSummary` takes a `logdir` argument, i.e., the directory where all of the data will be written. In addition, the `TFSummary` takes an optional argument `name` that specifies the name of event file. An `event` is a collection of summaries of scalars or images.  

```python
from tfsummary import TFSummary

tfs_training = TFSummary('path_to_log_dir', name='training')
tfs_validation = TFSummary('path_to_log_dir', name='validation')
```

### Logging with TFSummary

Logging data is primarily achieved via the high-level API: `TFSummary.add(global_step=None[, tag=value[, ...]])`. The argument `global_step` is generally used to specify the index of *step* or *epoch* during training. `TFSummary.add` takes variable number of keyword arguments in the format of `tag=value`:

```python
for i in xrange(1000):
    # iterative step, ex. gradient descent or evolutional step
    ...
    # log the progress every 100 steps
    if i % 100 == 0:
        tfs_training.add(global_step=i, error=training_err, loss=loss)
        tfs_validation.add(global_step=i, error=valid_error)
```
The `value` of a keyword argument in `TFSummary.add` can be a scalar or an "*image*". Note an "image" here can be a `matplotlib.Figure.figure` object or simply a 2-D `numpy` array. The latter will be rendered as a heatmap by
`matplotlib.pyplot.imshow`.

```python
import numpy
import matplotlib.pyplot as plt

noise = numpy.random.random((100, 100))
fig = plt.figure()
plt.plot(numpy.range(100), numpy.random.random(100))

tfs_validation.add(global_step=i, error=valid_error, result=fig, noise=noise)
```

### Launching TensorBoard

Follow the [instruction](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard#launching_tensorboard) on the official `Tensorflow` website for launching `TensorBoard`.

## Documentation

### Constructor

* `TFSummary`:

### High-Level API:

* `TFSummary.add`: add multiple scalars or images identified by different tags.

### Low-Level APIs

* `TFSummary.add_scalars`: add a single scalar.
* `TFSummary.add_scalar`: add multiple scalars under the same tag.
* `TFSummary.add_images`: add a single image.
* `TFSummary.add_image`: add multiple images under the same tag.

## Acknowledgements

Based on this [Gist](https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514) by [Michael Gygli
](https://github.com/gyglim).
