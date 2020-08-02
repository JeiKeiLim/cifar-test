import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa


class TFAugmentPolicy:
    def __init__(self, data_format="channels_last", resize=(None, None)):
        self.data_format = data_format
        self.resize = resize

        self.policy = [
            (flip_horizontal, 0.2),
            (color, 0.3),
            (rotate, 0.2),
            (rotate, 0.3)
        ]

    def __call__(self, x):
        if self.data_format == "channels_first":
            x = tf.transpose(x, perm=[1, 2, 0])

        for augment, p in self.policy:
            x = tf.cond(tf.random.uniform(shape=(), minval=0, maxval=1.0) < p,
                        lambda: augment(x), lambda: x)

        if self.resize[0] and self.resize[1]:
            x = tf.image.resize(x, self.resize)
        return x

    def __repr__(self):
        return "TFAugment DeepInAir AIHub KProducts Policy"


# Source from https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/
def flip_horizontal(x: tf.Tensor) -> tf.Tensor:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x)
    return x


def flip_vertical(x: tf.Tensor) -> tf.Tensor:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_up_down(x)
    return x


def color(x: tf.Tensor) -> tf.Tensor:
    """Color augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x


def rotate(x: tf.Tensor) -> tf.Tensor:
    """Rotation augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))


@tf.function
def rotate_tf(image):
    if image.shape.__len__() == 4:
        random_angles = tf.random.uniform(shape=(tf.shape(image)[0],), minval=-np.pi / 4, maxval=np.pi / 4)
    if image.shape.__len__() == 3:
        random_angles = tf.random.uniform(shape=(), minval=-np.pi / 4, maxval=np.pi / 4)

    return tfa.image.rotate(image, random_angles)


def zoom(x: tf.Tensor) -> tf.Tensor:
    """Zoom augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(32, 32))
        # Return a random crop
        return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]

    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))

