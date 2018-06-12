import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import numpy
import tensorflow as tf
from StringIO import StringIO

class Summary(object):
    """A Python wrapper of the Tensorflow Summary object.
    """
    def __init__(self, name, dirpath):
        self.name = name
        self.dirpath = dirpath
        self.filepath = os.path.join(self.dirpath, self.name)
        self.writer = tf.summary.FileWriter(self.filepath)

        self.summaries = []
        self.tf_summary = None

    def add(self, **kwargs):
        global_step = kwargs.pop('global_step', None)
        write = kwargs.pop('write', True)

        for key, val in kwargs.items():
            if hasattr(val, '__len__') and not isinstance(val, numpy.ndarray):
                self._add_multiple(key, val)
            else:
                self._add_single(key, val)

        if write:
            self._write_summary(global_step)

    def _add_single(self, tag, value):
        if numpy.isscalar(value):
            self.add_scalar(tag, value, write=False)
        elif isinstance(value, numpy.ndarray):
            self.add_image(tag, value, write=False)
        else:
            raise TypeError()

    def _add_multiple(self, tag, values):
        for i, value in enumerate(values):
            self._add_single('%s/%d' % (tag, i), value, write=False)

    def add_scalars(self, tag, scalars, write=True, global_step=None):
        for i, scalar in enumerate(scalars):
            self.add_scalar('%s/%d' % (tag, i), scalar, write=False)

        if write:
            self._write_summary(global_step)

    def add_scalar(self, tag, value, write=True, global_step=None):
        tf_summary_value = tf.Summary.Value(tag=tag, simple_value=value)
        self.summaries.append(tf_summary_value)

        if write:
            self._write_summary(global_step)

    def add_images(self, tag, images, write=True, global_step=None):
        for i, image in enumerate(images):
            self.add_image('%s/%d' % (tag, i), image, write=False)

        if write:
            self._write_summary(global_step)

    def add_image(self, tag, image, write=True, global_step=None):

        sio = StringIO()
        if isinstance(image, numpy.ndarray):
            plt.imsave(sio, image, format='png', dpi=300)
        else:
            raise TypeError()

        # Create an Image object
        tf_summary_image = tf.Summary.Image(
            encoded_image_string=sio.getvalue(),
            height=image.shape[0],
            width=image.shape[1])

        # Create a Summary value
        tf_summary_value = tf.Summary.Value(tag=tag, image=tf_summary_image)
        self.summaries.append(tf_summary_value)

        if write:
            self._write_summary(global_step)

    def add_histogram(self, write=True, global_step=None, **kwargs):

        if write:
            self._write_summary(global_step)

    def _write_summary(self, global_step=None):
        self.tf_summary = tf.Summary(value=self.summaries)
        self.writer.add_summary(self.tf_summary, global_step)
        del self.summaries[:]
        self.tf_summary = None
