import os.path
import random
import re
import shutil
import time
import zipfile
from glob import glob
from urllib.request import urlretrieve
from keras.utils import np_utils
from PIL import Image

import numpy as np
import scipy.misc
import tensorflow as tf
from imgaug import augmenters as iaa
from natsort import natsorted
from tqdm import tqdm


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def load_data(path):
    """load_data

    :param path: train or test directory path

    """

    train_img_path = path + '/training/train/'
    train_masks_path = path + '/training/train_masks/'

    train_img_list = sorted(glob(train_img_path + '*'))
    train_masks_list = sorted(glob(train_masks_path + '*'))

    # test_img_path = path + 'test/pos/'
    # test_box_path = path + 'test/posGt/'

    # test_img_list = sorted(glob.glob(test_img_path + '*'))
    # test_box_list = sorted(glob.glob(test_box_path + '*'))

    return train_img_list, train_masks_list


class Datagen(object):
    def __init__(self, img_list, mask_list, img_shape, batch_size, num_classes,
                 shuffle=True, augment=True, verbose=False):
        """Data Generator for Semantic Segmentation

        :param img_list: list of image file path
        :param mask_list: list of mask file path
        :param img_shape: Shape of a image file in tuple EX:(120, 560, 3)
        :param batch_size: size of a batch
        :param shuffle: To shuffle during data generation or not
        :param verbose: Print out verbose for debugging
        :returns: data and labels for training
        :rtype: data is (batch_size, height, width, channels)
                label is (batch_size, height, width, label)

        """
        self.img_list = img_list
        self.mask_list = mask_list
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augment = augment
        self.verbose = verbose
        self.max_index = len(img_list)
        self.batch_start = 0
        self.batch_iter = 0
        self.batch_end = 0
        self.batch_rows = None
        self.truth_color = np.array([[255]])

    def generate_train_batch(self):
        """Generate training data in batch

        :returns: data and labels
        :rtype: data is (batch_size, height, width, channels)
                label is (batch_size, height, width, label)

        """

        while True:

            if self.shuffle:

                self.batch_rows = np.random.randint(0, self.max_index, self.batch_size)

            else:

                self.batch_end = self.batch_start + self.batch_size

                if self.batch_end >= self.max_index:
                    remainder = self.batch_end - self.max_index
                    self.batch_rows = np.concatenate((np.arange(self.batch_start, self.max_index), np.arange(0, remainder)))
                    self.batch_start = remainder
                else:
                    self.batch_rows = np.arange(self.batch_start, self.batch_start + self.batch_size)
                    self.batch_start += self.batch_size

            batch_images = np.empty((self.batch_size, self.img_shape[0], self.img_shape[1], self.img_shape[2]), dtype='uint8')
            batch_masks = np.empty((self.batch_size, self.img_shape[0], self.img_shape[1]))

            batch_images[np.arange(self.batch_size)] = [np.asarray(Image.open(self.img_list[ind])) for ind in self.batch_rows]
            batch_masks[np.arange(self.batch_size)] = [np.asarray(Image.open(self.mask_list[ind])) for ind in self.batch_rows]
            batch_masks = np_utils.to_categorical(batch_masks)

            yield batch_images, batch_masks


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """

        # Load datasets into list
        image_paths, mask_paths = load_data(data_folder)

        # Designate the background color
        background_color = np.array([0])

        flipper = iaa.Fliplr(1.0) # always horizontally flip each input image
        vflipper = iaa.Flipud(1.0) # vertically flip each input image with 90% probability
        # blurer = iaa.GaussianBlur(3.0) # apply gaussian blur
        blurer = iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
        lighter = iaa.Add((-10, 10), per_channel=0.5) # change brightness of images (by -10 to 10 of original value)
        translater = iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}) # translate by -20 to +20 percent (per axis)

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file),
                                            image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file),
                                               image_shape)

                # if np.random.random() > 0.7:
                #     image = flipper.augment_image(image)
                #     gt_image = flipper.augment_image(gt_image)
                # if np.random.random() > 0.7:
                #     image = vflipper.augment_image(image)
                #     gt_image = vflipper.augment_image(gt_image)
                # # if np.random() > 0.7:
                # #     image = translater(image)
                # #     gt_image = translater(gt_image)

                # if np.random.random() > 0.7:
                #     image = blurer.augment_image(image)
                # if np.random.random() > 0.7:
                #     image = lighter.augment_image(image)

                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder,
                    image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0],
                                                 image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0],
                                                  image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape,
                           logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
