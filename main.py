"""
Module to construct tf graph for semantic segmentation
"""

import os.path
import warnings
from distutils.version import LooseVersion
import tensorflow as tf
import helper
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), \
    'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(
        tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural \
                  network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# Global Variables

NUM_CLASSES = 2
WEIGHT_INIT_STD = 0.01
WEIGHT_L2 = 1e-3
IMAGE_SHAPE = (1280, 1918)

DATA_DIRECTORY = './data'
RUNS_DIRECTORY = './runs'


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and
                     "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob,
             layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    image_input = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out


def conv_1x1(layer, kernel=1, strides=1):
    """conv_1x1
    Create 1x1 conlution of a given layer

    :param layer: TF Tensor
    :param kernel: kernel size which would be 1
    :param strides: strides
    """
    return tf.layers.conv2d(inputs=layer, filters=NUM_CLASSES,
                            kernel_size=kernel, strides=strides,
                            padding='same',
                            kernel_initializer=tf.random_normal_initializer(stddev=WEIGHT_INIT_STD),
                            kernel_regularizer = tf.contrib.layers.l2_regularizer(WEIGHT_L2))


def conv_upsample(layer, kernel=4, strides=2):
    """conv_upsample
    Create upsampling aka.transpose convolution

    :param layer: TF Tensor
    :param kernel: kernel size
    :param strides: strides
    """
    return tf.layers.conv2d_transpose(inputs=layer, filters=NUM_CLASSES,
                                      kernel_size=kernel, strides=strides,
                                      padding='same',
                                      kernel_initializer=tf.random_normal_initializer(stddev=WEIGHT_INIT_STD),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_L2))


def skip_connection(layer_1, layer_2):
    """skip_connection

    :param layer_1: Layer to connect from
    :param layer_2: Layer to connect to
    """
    return tf.add(layer_1, layer_2)


def layers(layer_3, layer_4, layer_7, num_classes=NUM_CLASSES):
    """
    Create the layers for a fully convolutional network.
    Build skip-layers using the vgg layers.
    :param layer_7: TF Tensor for VGG Layer 3 output
    :param layer_4: TF Tensor for VGG Layer 4 output
    :param layer_3: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    layer_7_conv = conv_1x1(layer_7)
    layer_7_upsample = conv_upsample(layer_7_conv)

    layer_4_conv = conv_1x1(layer_4)
    skip_7_4 = skip_connection(layer_7_upsample, layer_4_conv)
    layer_4_upsample = conv_upsample(skip_7_4)

    layer_3_conv = conv_1x1(layer_3)
    skip_4_3 = skip_connection(layer_4_upsample, layer_3_conv)
    layer_3_upsample = conv_upsample(skip_4_3, kernel=16, strides=8)

    return layer_3_upsample


def optimize(nn_last_layer, correct_label, learning_rate,
             num_classes=NUM_CLASSES):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, NUM_CLASSES))
    correct_label = tf.reshape(correct_label, (-1, NUM_CLASSES))

    # define loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    # define training operation
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
             cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.
                           Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    sess.run(tf.global_variables_initializer())

    print("Training Started...")
    print("------------")

    for i in range(epochs):

        print("EPOCH {} ...".format(i+1))

        for image, label in get_batches_fn(batch_size):

            feed_dict = {input_image: image,
                         correct_label: label,
                         keep_prob: 0.5,
                         learning_rate: 0.0009}

            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict=feed_dict)

            print("Loss: = {:.3f}".format(loss))
        print("-------------")
    print("Training Done!")


def run():
    """run
    Run the pipeline
    """

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(DATA_DIRECTORY)

    # OPTIONAL: Train and Inference on the cityscapes dataset
    #           instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:

        # Path to vgg model
        vgg_path = os.path.join(DATA_DIRECTORY, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(DATA_DIRECTORY, 'data/training'), IMAGE_SHAPE)

        # OPTIONAL: Augment Images for better results
        # Build NN using load_vgg, layers, and optimize function

        epochs = 10
        batch_size = 5

        # TF placeholders
        correct_label = tf.placeholder(tf.int32,
                                       [None, None, None, NUM_CLASSES],
                                       name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out,
                               vgg_layer7_out, NUM_CLASSES)

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer,
                                                        correct_label,
                                                        learning_rate,
                                                        NUM_CLASSES)

        # Train NN using the train_nn function

        train_nn(sess, epochs, batch_size, get_batches_fn,
                 train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(RUNS_DIRECTORY,
                                      DATA_DIRECTORY,
                                      sess, IMAGE_SHAPE, logits,
                                      keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


def run_tests():
    """run_tests
    run tests
    """

    tests.test_load_vgg(load_vgg, tf)
    tests.test_layers(layers)
    tests.test_optimize(optimize)
    tests.test_for_kitti_dataset(DATA_DIRECTORY)
    tests.test_train_nn(train_nn)


if __name__ == '__main__':
    run_tests()
    run()
