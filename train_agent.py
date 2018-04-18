import shutil
import os
import sys
import time

import tensorflow as tf
import numpy as np

from model_file import RNN_Model
from prepare_data import load_data
from aux_code.ops import (get_minibatches_indices, pad_seqs,
                          print_metrics,
                          get_max_length, calculate_metrics, create_conf_dict,
                          save_config_dict, get_validset_feeds)
from aux_code.tf_ops import create_scalar_summaries, create_sess


class TrainAgent(object):
    def __init__(self, args):
        # Create the config dictionary
        self.config = create_conf_dict(args)

        self.logdir = os.path.join(args.logdir,args.name)
        if os.path.exists(self.logdir):
            shutil.rmtree(self.logdir)

        self.h_dim = list(map(int, args.layers.split(',')))

        # Create the save path
        self.save_path = 'models/' + args.name + '/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.mse = False
        if args.data == 'add':
            self.mse = True

        self.output_format = 'last'
        if args.data == 'copy':
            self.output_format = 'all'

        self.log_test = False
        if args.log_test or args.data == 'copy' or args.data == 'add':
            self.log_test = True

        if args.cell == 'janet':
            args.chrono = True

    def train(self, data_path, max_gradient_norm, weight_decay, test_agent,
              args):
        data_list = load_data(data_path, seq_len=args.T)
        x_train, y_train, x_valid, y_valid, x_test, y_test = data_list

        max_len = get_max_length(x_train + x_valid)

        # Create the model
        model = RNN_Model(x_train[0].shape[-1], y_train.shape[-1],
                          h_dim=self.h_dim,
                          max_sequence_length=max_len,
                          max_gradient_norm=max_gradient_norm,
                          opt_method=args.optimizer,
                          weight_decay=weight_decay,
                          cell_type=args.cell,
                          chrono=args.chrono,
                          mse=self.mse,
                          )
        model.build(self.output_format)

        saver = tf.train.Saver()

        # Prepare validation and test data
        v_input_feed, v_output_feed = get_validset_feeds(
            model, x_valid, y_valid, self.h_dim)
        t_input_feed, t_output_feed = get_validset_feeds(
            model, x_test, y_test, self.h_dim)

        train_tags = ['Train Loss', 'Train Acc', 'Train F1']
        metric_tags = train_tags + ['Valid Loss', 'Valid Acc', 'Valid F1']

        if self.log_test:
            metric_tags += ['Test Loss', 'Test Acc', 'Test F1']

        with create_sess() as sess:
            sess.run([tf.global_variables_initializer(),
                      tf.local_variables_initializer()])
            best_loss = 1e8
            best_epoch = 0

            tb_writer = tf.summary.FileWriter(self.logdir, sess.graph)

            print("Training model ...")
            tb_step = 0
            for e in range(args.epochs):
                minibatch_indices = get_minibatches_indices(
                    len(x_train), args.batch_size)

                eval_metrics = np.zeros(len(metric_tags))

                # Time each epoch
                start_time = time.time()

                # Loop over minibatches
                for b_num, b_indices in enumerate(minibatch_indices):
                    print('\rProcessing batch {}/{}'.format(
                        b_num, len(minibatch_indices)), end='', flush=True)

                    x = [x_train[i] for i in b_indices]
                    y = y_train[b_indices]

                    x, seq_lengths = pad_seqs(x)

                    input_feed = {model.x: x,
                                  model.y: y,
                                  model.seq_lens: seq_lengths,
                                  model.training: True,
                                  model.keep_prob: args.keep_prob,
                                  }
                    _, loss, output_probs = sess.run(
                        [model.train_opt, model.loss_nowd,
                         model.output_probs],
                        input_feed)

                    if np.isnan(loss):
                        print('!'*70)
                        print('Nan loss value')
                        sys.exit()

                    # Update the training loss
                    eval_metrics[0] += loss / float(len(minibatch_indices))

                    if b_num % args.log_every == 0:
                        summary = create_scalar_summaries(
                            ['high_res_train_loss'],
                            [loss],
                        )

                        tb_writer.add_summary(summary, tb_step)
                        tb_writer.flush()
                        tb_step += 1

                    # Update the remaining metrics
                    eval_metrics[1:3] += calculate_metrics(
                        y, np.argmax(output_probs, axis=1)) / \
                        float(len(minibatch_indices))

                if 'mnist' in args.data:
                    # Compute validation loss and accuracy
                    eval_metrics[len(train_tags)], output_probs = sess.run(
                        v_output_feed, v_input_feed)
                    eval_metrics[len(train_tags)+1:2*len(train_tags)] += \
                        calculate_metrics(y_valid,
                                          np.argmax(output_probs, axis=1))

                    epoch_duration = time.time() - start_time

                    if self.log_test:
                        # Compute test dataset metrics
                        eval_metrics[2*len(train_tags)], output_probs =\
                            sess.run(t_output_feed, t_input_feed)
                        eval_metrics[2*len(train_tags)+1:] += calculate_metrics(
                            y_test, np.argmax(output_probs, axis=1))

                    # Log the metrics on tensorboard
                    summary = create_scalar_summaries(
                        metric_tags + ['Epoch_duration'],
                        np.concatenate((eval_metrics, [epoch_duration])))

                    tb_writer.add_summary(summary, e)

                    # Save model if it yields the best validation loss
                    if best_loss >= eval_metrics[len(train_tags)]:
                        best_loss = eval_metrics[len(train_tags)]
                        best_epoch = e
                        saver.save(sess, self.save_path + 'model')

                        # Save the config to json
                        save_config_dict(self.config, self.save_path,
                                         tags=['best_epoch',
                                               'max_sequence_length'] +
                                         metric_tags,
                                         values=[best_epoch, max_len] +
                                         list(eval_metrics))

                    print()
                    print(args.name + ": Epoch {}/{}  |  best epoch: {}"
                          "  |  epoch duration: {:.1f}".format(
                              e, args.epochs, best_epoch, epoch_duration))
                    print_metrics(metric_tags, eval_metrics)

                else:
                    save_config_dict(self.config, self.save_path,
                                     tags=['placeholder'],
                                     values=[0],
                                     )

                    print()
                    print(args.name + ": Epoch {}/{}".format(e, args.epochs))

        if 'mnist' in args.data:
            test_agent.test(x_test, y_test, self.save_path, self.config)
