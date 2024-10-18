from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tensorflow as tf

from modules.nextsrgan import RRDB_Model
from modules.lr_scheduler import MultiStepLR
from modules.losses import PixelLoss
from modules.utils import (load_yaml, load_dataset, ProgressBar,
                           set_memory_growth)

flags.DEFINE_string('cfg_path', './configs/psnr.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')

def main(_):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    model = RRDB_Model(cfg['input_size'], cfg['ch_size'], cfg['network_G'])
    model.summary(line_length=80)

    train_dataset = load_dataset(cfg, 'train_dataset', shuffle=True)

    learning_rate = MultiStepLR(cfg['lr'], cfg['lr_steps'], cfg['lr_rate'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         beta_1=cfg['adam_beta1_G'],
                                         beta_2=cfg['adam_beta2_G'])

    pixel_loss_fn = PixelLoss(criterion=cfg['pixel_criterion'])

    checkpoint_dir = './checkpoints/' + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name='step'),
                                     optimizer=optimizer,
                                     model=model)
    manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                         directory=checkpoint_dir,
                                         max_to_keep=3)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print(f'[*] load ckpt from {manager.latest_checkpoint} at step {checkpoint.step.numpy()}.')
    else:
        print("[*] training from scratch.")

    @tf.function
    def train_step(lr, hr):
        with tf.GradientTape() as tape:
            sr = model(lr, training=True)

            losses = {}
            losses['reg'] = tf.reduce_sum(model.losses)
            losses['pixel'] = cfg['w_pixel'] * pixel_loss_fn(hr, sr)
            total_loss = tf.add_n([l for l in losses.values()])

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return total_loss, losses

    summary_writer = tf.summary.create_file_writer('./logs/' + cfg['sub_name'])
    prog_bar = ProgressBar(cfg['niter'], checkpoint.step.numpy())
    remain_steps = max(cfg['niter'] - checkpoint.step.numpy(), 0)

    for lr, hr in train_dataset.take(remain_steps):
        checkpoint.step.assign_add(1)
        steps = checkpoint.step.numpy()

        total_loss, losses = train_step(lr, hr)

        prog_bar.update(f"loss={total_loss.numpy():.4f}, lr={optimizer.lr(steps).numpy():.1e}")

        if steps % 10 == 0:
            with summary_writer.as_default():
                tf.summary.scalar('loss/total_loss', total_loss, step=steps)
                for k, l in losses.items():
                    tf.summary.scalar(f'loss/{k}', l, step=steps)
                tf.summary.scalar('learning_rate', optimizer.lr(steps), step=steps)

        if steps % cfg['save_steps'] == 0:
            manager.save()
            print(f"\n[*] save ckpt file at {manager.latest_checkpoint}")

    print("\n[*] training done!")

if __name__ == '__main__':
    app.run(main)