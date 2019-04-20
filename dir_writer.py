import os
import numpy as np
from multiprocessing import pool
from utils import _list_valid_filenames_in_directory
from writer import Writer


class DirectoryWriter(Writer):
    """Writing images and labels  into tfrecord after reading them from folder
    
            directory: string, path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`.
            Color mode to read images.
        classes: Optional list of strings, names of subdirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
            If set to False, sorts the data in alphanumeric order.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
        tfrecord: the tfrecord file name where to store the data
        num_copies: the number of random generated image from one sample
        dtype: Dtype to use for generated arrays.
    
    Returns:
        [type] -- [description]
    """
    
    allowed_class_modes = {'categorical', 'binary', 'sparse', 'input', None}

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


