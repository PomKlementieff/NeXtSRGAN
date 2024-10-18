import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input, VGG19


def PixelLoss(criterion='l1'):
    if criterion == 'l1':
        return tf.keras.losses.MeanAbsoluteError()
    elif criterion == 'l2':
        return tf.keras.losses.MeanSquaredError()
    else:
        raise NotImplementedError(
            'Loss type {} is not recognized.'.format(criterion))


def ContentLoss(criterion='l1', output_layer=54, before_act=True):
    if criterion == 'l1':
        loss_func = tf.keras.losses.MeanAbsoluteError()
    elif criterion == 'l2':
        loss_func = tf.keras.losses.MeanSquaredError()
    else:
        raise NotImplementedError(
            'Loss type {} is not recognized.'.format(criterion))
    vgg = VGG19(input_shape=(None, None, 3), include_top=False)

    if output_layer == 22:
        pick_layer = 5
    elif output_layer == 54:
        pick_layer = 20
    else:
        raise NotImplementedError(
            'VGG output layer {} is not recognized.'.format(criterion))

    if before_act:
        vgg.layers[pick_layer].activation = None

    fea_extrator = tf.keras.Model(vgg.input, vgg.layers[pick_layer].output)

    @tf.function
    def content_loss(hr, sr):
        preprocess_sr = preprocess_input(sr * 255.) / 12.75
        preprocess_hr = preprocess_input(hr * 255.) / 12.75
        sr_features = fea_extrator(preprocess_sr)
        hr_features = fea_extrator(preprocess_hr)

        return loss_func(hr_features, sr_features)

    return content_loss


def DiscriminatorLoss(gan_type='ragan'):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    sigma = tf.sigmoid

    def discriminator_loss_ragan(hr, sr):
        return 0.5 * (
            cross_entropy(tf.ones_like(hr), sigma(hr - tf.reduce_mean(sr))) +
            cross_entropy(tf.zeros_like(sr), sigma(sr - tf.reduce_mean(hr))))

    def discriminator_loss(hr, sr):
        real_loss = cross_entropy(tf.ones_like(hr), sigma(hr))
        fake_loss = cross_entropy(tf.zeros_like(sr), sigma(sr))
        return real_loss + fake_loss

    if gan_type == 'ragan':
        return discriminator_loss_ragan
    elif gan_type == 'gan':
        return discriminator_loss
    else:
        raise NotImplementedError(
            'Discriminator loss type {} is not recognized.'.format(gan_type))


def GeneratorLoss(gan_type='ragan'):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    sigma = tf.sigmoid

    def generator_loss_ragan(hr, sr):
        return 0.5 * (
            cross_entropy(tf.ones_like(sr), sigma(sr - tf.reduce_mean(hr))) +
            cross_entropy(tf.zeros_like(hr), sigma(hr - tf.reduce_mean(sr))))

    def generator_loss(hr, sr):
        return cross_entropy(tf.ones_like(sr), sigma(sr))

    if gan_type == 'ragan':
        return generator_loss_ragan
    elif gan_type == 'gan':
        return generator_loss
    else:
        raise NotImplementedError(
            'Generator loss type {} is not recognized.'.format(gan_type))