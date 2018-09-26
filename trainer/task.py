import argparse
import os

import trainer.model as model

import tensorflow as tf
from tensorflow.contrib.learn import learn_runner
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
from tensorflow.contrib.training.python.training import hparam

tfgan = tf.contrib.gan
tf.logging.set_verbosity(tf.logging.ERROR)


def run_experiment(hparams):
    """Run the training and evaluate using the high level API"""
    train_input_func = tf.estimator.inputs.numpy_input_fn(model.input_func(hparams.train_noisy_files),
                                                          model.input_func(hparams.train_clean_files),
                                                          batch_size=hparams.train_batch_size,
                                                          num_epochs=100,
                                                          shuffle=True)

    eval_input_func = tf.estimator.inputs.numpy_input_fn(model.input_func(hparams.eval_noisy_files),
                                                         model.input_func(hparams.eval_clean_files),
                                                         batch_size=hparams.eval_batch_size,
                                                         num_epochs=100,
                                                         shuffle=True)

    denoising_gan = tfgan.estimator.GANEstimator(model_dir=hparams.job_dir,
                                                 generator_fn=model.generator_fn,
                                                 discriminator_fn=model.discriminator_fn,
                                                 generator_loss_fn=tfgan.losses.minimax_generator_loss,
                                                 discriminator_loss_fn=tfgan.losses.minimax_discriminator_loss,
                                                 generator_optimizer=tf.train.AdamOptimizer(hparams.learning_rate, beta1=0.5),
                                                 discriminator_optimizer=tf.train.AdamOptimizer(hparams.learning_rate, beta1=0.5),
                                                 add_summaries=tfgan.estimator.SummaryType.VARIABLES,
                                                 use_loss_summaries=True
                                                )

    denoising_gan.train(train_input_func, steps=hparams.train_steps)
    denoising_gan.evaluate(eval_input_func)


    #add export_savedmodel

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  if __name__ == '__main__':
      parser = argparse.ArgumentParser()
      # Input Arguments
      parser.add_argument(
          '--train-clean-files',
          help='GCS or local paths to training data',
          nargs='+',
          required=True
      )
      parser.add_argument(
          '--train-noisy-files',
          help='GCS or local paths to training data',
          nargs='+',
          required=True
      )
      parser.add_argument(
          '--num-epochs',
          help="""\
        Maximum number of training data epochs on which to train.
        If both --max-steps and --num-epochs are specified,
        the training job will run for --max-steps or --num-epochs,
        whichever occurs first. If unspecified will run for --max-steps.\
        """,
          type=int,
      )
      parser.add_argument(
          '--train-batch-size',
          help='Batch size for training steps',
          type=int,
          default=32
      )
      parser.add_argument(
          '--eval-batch-size',
          help='Batch size for evaluation steps',
          type=int,
          default=40
      )
      parser.add_argument(
          '--eval-noisy-files',
          help='GCS or local paths to evaluation data',
          nargs='+',
          required=True
      )
      parser.add_argument(
          '--eval-clean-files',
          help='GCS or local paths to evaluation data',
          nargs='+',
          required=True
      )
      # Training arguments

      parser.add_argument(
          '--learning-rate',
          help='Learning rate for the optimizer',
          default=0.001,
          type=float
      )

      parser.add_argument(
          '--job-dir',
          help='GCS location to write checkpoints and export models',
          required=True
      )

      # Experiment arguments
      parser.add_argument(
          '--train-steps',
          help="""\
        Steps to run the training job for. If --num-epochs is not specified,
        this must be. Otherwise the training job will run indefinitely.\
        """,
          type=int
      )

      args = parser.parse_args()

      # Run the training job
      hparams = hparam.HParams(**args.__dict__)
      run_experiment(hparams)
