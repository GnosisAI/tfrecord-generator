from __future__ import absolute_import, division, print_function

import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils import array_to_img, img_to_array, load_img


class Writer():
    """Writes all images and labels to trfecord file after augmentation.
    """

    def set_processing_attrs(self,
                             image_data_generator,
                             target_size,
                             color_mode,
                             data_format,
                             interpolation,
                             tfrecord,
                             num_copies):
        """Sets attributes to use later for processing files into a batch.
        # Arguments
            image_data_generator: Instance of `ImageDataGenerator`
                to use for random transformations and normalization.
            target_size: tuple of integers, dimensions to resize input images to.
            color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`.
                Color mode to read images.
            data_format: String, one of `channels_first`, `channels_last`.
            interpolation: Interpolation method used to resample the image if the
                target size is different from that of the loaded image.
                Supported methods are "nearest", "bilinear", and "bicubic".
                If PIL version 1.1.3 or newer is installed, "lanczos" is also
                supported. If PIL version 3.4.0 or newer is installed, "box" and
                "hamming" are also supported. By default, "nearest" is used.
            tfrecord: String, the tfrecord file name to store data
            num_copies: the number of random generated image from one sample
        """
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'rgba', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb", "rgba", or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgba':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (4,)
            else:
                self.image_shape = (4,) + self.target_size
        elif self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size

        self.interpolation = interpolation
        self.tfrecord = tfrecord
        self.num_copies = num_copies

    def write_tfrecord(self):
        """writes data to tfrecord file
        """
        # build batch of image data
        # self.filepaths is dynamic, is better to call it once outside the loop
        filepaths = self.filepaths
        labels = self.labels
        tfrecord = self.tfrecord
        with tf.io.TFRecordWriter(tfrecord) as writer:
            for fpath, label in tqdm(zip(filepaths, labels), desc='writing images to tfrecords'):
                img = load_img(fpath,
                               color_mode=self.color_mode,
                               target_size=self.target_size,
                               interpolation=self.interpolation)
                x = img_to_array(img, data_format=self.data_format)
                # Pillow images should be closed after `load_img`,
                # but not PIL images.
                if hasattr(img, 'close'):
                    img.close()
                if self.image_data_generator:
                    for _ in range(self.num_copies):
                        x_copy = x.copy()
                        params = self.image_data_generator.get_random_transform(
                            x_copy.shape)
                        x_copy = self.image_data_generator.apply_transform(
                            x_copy, params)
                        x_copy = self.image_data_generator.standardize(x_copy)
                        # convert augmented image
                        self._write_image(x_copy, label, writer)

                # write th original
                self._write_image(x, label, writer)

    def _write_image(self, img_arr, label, writer):
        img_bin = img_arr.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bin])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))

        writer.write(example.SerializeToString())
