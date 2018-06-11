import os
import tensorflow as tf

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
        step = kwargs.pop('global_step', None)

        for key, val in kwargs.items():
            if not hasattr(val, '__len__'):
                self.add_scalar(key, val, write=False)

        self._write_summary(global_step)

    def add_scalar(self, tag, value, write=True, global_step=None):
        tf_summary_value = tf.Summary.Value(tag=tag, simple_value=value)
        self.summaries.append(tf_summary_value)

        if write:
            self._write_summary(global_step)

    def add_images(self, write=True, global_step=None, **kwargs):

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
