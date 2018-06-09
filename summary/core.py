import os
import tensorflow as tf

class Summary(object):
    """A Python wrapper of the Tensorflow Summary object.
    """
    def __init__(self, name, dirpath, *args, **kwargs):
        self.name = name
        self.dirpath = dirpath
        self.filepath = os.path.join(self.dirpath, self.name)
        self.writer = tf.summary.FileWriter(self.filepath)

        self.attrs = dict()
        self.tf_summary = tf.Summary()

        for arg in args:
            self.attrs[arg] = len(self.attrs)
            self.tf_summary.value.add(tag=arg, simple_value=None)

        for key, val in kwargs.items():
            self.attrs[key] = len(self.attrs)
            self.tf_summary.value.add(tag=key, simple_value=val)

    def add(self, **kwargs):
        step = kwargs.pop('global_step', None)

        for key, val in kwargs.items():
            index = self.attrs.get(key)
            if index is not None:
                self.tf_summary.value[index].simple_value = val

        self.writer.add_summary(self.tf_summary, step)
