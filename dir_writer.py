import os
import numpy as np
from multiprocessing import pool
from utils import _list_valid_filenames_in_directory
from writer import Writer


class DirectoryWriter(Writer):
    def __init__(self,
                 directory,
                 image_data_generator=None,
                 target_size=(256, 256),
                 color_mode='rgb',
                 classes=None,
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 data_format='channels_last',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest',
                 tfrecord='dataset.tfr',
                 num_copies=3,
                 dtype='float32'):
        self.set_processing_attrs(image_data_generator,
                                  target_size,
                                  color_mode,
                                  data_format,
                                  interpolation,
                                  tfrecord,
                                  num_copies)

        self.directory = directory
        self.white_list_formats = 'png'

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        p = pool.ThreadPool()
        # Second, build an index of the images
        # in the different class subfolders.
        results = []
        self.filenames = []
        i = 0
        for dirpath in (os.path.join(directory, subdir) for subdir in classes):
            results.append(
                p.apply_async(_list_valid_filenames_in_directory,
                              (dirpath, self.white_list_formats,
                                  self.class_indices, False)))
        classes_list = []
        for res in results:
            classes, filenames = res.get()
            classes_list.append(classes)
            self.filenames += filenames
        self.samples = len(self.filenames)
        self.classes = np.zeros((self.samples,), dtype='int32')
        for classes in classes_list:
            self.classes[i:i + len(classes)] = classes
            i += len(classes)

        print('Found %d images belonging to %d classes.' %
              (self.samples, self.num_classes))
        p.close()
        p.join()
        self._filepaths = [
            os.path.join(self.directory, fname) for fname in self.filenames
        ]

    @property
    def filepaths(self):
        return self._filepaths

    @property
    def labels(self):
        return self.classes


