from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tensorflow as tf

from modules.nextsrgan import RRDB_Model, DiscriminatorVGG128
from modules.lr_scheduler import MultiStepLR
from modules.losses import (PixelLoss, ContentLoss, DiscriminatorLoss,
                            GeneratorLoss)
from modules.utils import (load_yaml, load_dataset, ProgressBar,
                           set_memory_growth)

flags.DEFINE_string('cfg_path', './configs/nextsrgan.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')

def main(_):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    generator = RRDB_Model(cfg['input_size'], cfg['ch_size'], cfg['network_G'])
    generator.summary(line_length=80)
    discriminator = DiscriminatorVGG128(cfg['gt_size'], cfg['ch_size'])
    discriminator.summary(line_length=80)

    train_dataset = load_dataset(cfg, 'train_dataset', shuffle=False)

    learning_rate_G = MultiStepLR(cfg['lr_G'], cfg['lr_steps'], cfg['lr_rate'])
    learning_rate_D = MultiStepLR(cfg['lr_D'], cfg['lr_steps'], cfg['lr_rate'])
    optimizer_G = tf.keras.optimizers.Adam(learning_rate=learning_rate_G,
                                           beta_1=cfg['adam_beta1_G'],
                                           beta_2=cfg['adam_beta2_G'])
    optimizer_D = tf.keras.optimizers.Adam(learning_rate=learning_rate_D,
                                           beta_1=cfg['adam_beta1_D'],
                                           beta_2=cfg['adam_beta2_D'])

    pixel_loss_fn = PixelLoss(criterion=cfg['pixel_criterion'])
    fea_loss_fn = ContentLoss(criterion=cfg['feature_criterion'])
    gen_loss_fn = GeneratorLoss(gan_type=cfg['gan_type'])
    dis_loss_fn = DiscriminatorLoss(gan_type=cfg['gan_type'])

    checkpoint_dir = './checkpoints/' + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name='step'),
                                     optimizer_G=optimizer_G,
                                     optimizer_D=optimizer_D,
                                     model=generator,
                                     discriminator=discriminator)
    manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                         directory=checkpoint_dir,
                                         max_to_keep=3)

    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print(f'[*] load ckpt from {manager.latest_checkpoint} at step {checkpoint.step.numpy()}.')
    else:
        if cfg['pretrain_name'] is not None:
            pretrain_dir = './checkpoints/' + cfg['pretrain_name']
            if tf.train.latest_checkpoint(pretrain_dir):
                checkpoint.restore(tf.train.latest_checkpoint(pretrain_dir))
                checkpoint.step.assign(0)
                print(f"[*] training from pretrain model {pretrain_dir}.")
            else:
                print(f"[*] cannot find pretrain model {pretrain_dir}.")
        else:
            print("[*] training from scratch.")

    @tf.function
    def train_step(lr, hr):
        with tf.GradientTape(persistent=True) as tape:
            sr = generator(lr, training=True)
            hr_output = discriminator(hr, training=True)
            sr_output = discriminator(sr, training=True)

            losses_G = {}
            losses_D = {}
            losses_G['reg'] = tf.reduce_sum(generator.losses)
            losses_D['reg'] = tf.reduce_sum(discriminator.losses)
            losses_G['pixel'] = cfg['w_pixel'] * pixel_loss_fn(hr, sr)
            losses_G['feature'] = cfg['w_feature'] * fea_loss_fn(hr, sr)
            losses_G['gan'] = cfg['w_gan'] * gen_loss_fn(hr_output, sr_output)
            losses_D['gan'] = dis_loss_fn(hr_output, sr_output)
            total_loss_G = tf.add_n([l for l in losses_G.values()])
            total_loss_D = tf.add_n([l for l in losses_D.values()])

        grads_G = tape.gradient(total_loss_G, generator.trainable_variables)
        grads_D = tape.gradient(total_loss_D, discriminator.trainable_variables)
        optimizer_G.apply_gradients(zip(grads_G, generator.trainable_variables))
        optimizer_D.apply_gradients(zip(grads_D, discriminator.trainable_variables))

        return total_loss_G, total_loss_D, losses_G, losses_D

    summary_writer = tf.summary.create_file_writer('./logs/' + cfg['sub_name'])
    prog_bar = ProgressBar(cfg['niter'], checkpoint.step.numpy())
    remain_steps = max(cfg['niter'] - checkpoint.step.numpy(), 0)

    for lr, hr in train_dataset.take(remain_steps):
        checkpoint.step.assign_add(1)
        steps = checkpoint.step.numpy()

        total_loss_G, total_loss_D, losses_G, losses_D = train_step(lr, hr)

        prog_bar.update(
            f"loss_G={total_loss_G.numpy():.4f}, loss_D={total_loss_D.numpy():.4f}, "
            f"lr_G={optimizer_G.lr(steps).numpy():.1e}, lr_D={optimizer_D.lr(steps).numpy():.1e}")

        if steps % 10 == 0:
            with summary_writer.as_default():
                tf.summary.scalar('loss_G/total_loss', total_loss_G, step=steps)
                tf.summary.scalar('loss_D/total_loss', total_loss_D, step=steps)
                for k, l in losses_G.items():
                    tf.summary.scalar(f'loss_G/{k}', l, step=steps)
                for k, l in losses_D.items():
                    tf.summary.scalar(f'loss_D/{k}', l, step=steps)

                tf.summary.scalar('learning_rate_G', optimizer_G.lr(steps), step=steps)
                tf.summary.scalar('learning_rate_D', optimizer_D.lr(steps), step=steps)

        if steps % cfg['save_steps'] == 0:
            manager.save()
            print(f"\n[*] save ckpt file at {manager.latest_checkpoint}")

    print("\n [*] training done!")

if __name__ == '__main__':
    app.run(main)