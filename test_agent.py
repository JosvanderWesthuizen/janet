import csv
import os
import time

import numpy as np
import tensorflow as tf

from model_file import RNN_Model
from aux_code.ops import (print_metrics,
                          get_max_length, calculate_metrics,
                          save_config_dict, get_validset_feeds)
from aux_code.tf_ops import create_sess


class TestAgent(object):
    def __init__(self, args):
        self.h_dim = list(map(int, args.layers.split(',')))
        self.batch_size = args.batch_size
        self.keep_prob = args.keep_prob

        self.name = args.name

        self.args = vars(args)

        self.mse = False
        if args.data == 'add':
            self.mse = True

        self.output_format = 'last'
        if args.data == 'copy':
            self.output_format = 'all'

    def test(self, x, y, model_path, conf_dict=None):
        tf.reset_default_graph()
        max_len = get_max_length(x)

        # Create the model
        model = RNN_Model(x[0].shape[-1], y.shape[-1],
                          h_dim=self.h_dim,
                          max_sequence_length=max_len,
                          is_test=True,
                          cell_type=self.args['cell'],
                          mse=self.mse,
                          )
        model.build(self.output_format)

        print('Variables to be loaded')
        for v in tf.trainable_variables():
            print(v)

        # Prepare the test data
        input_feed, output_feed = get_validset_feeds(
            model, x, y, self.h_dim)

        metric_tags = ['Test Loss', 'Test Acc', 'Test F1']

        with create_sess() as sess:
            saver = tf.train.Saver()
            saver.restore(sess,
                          tf.train.latest_checkpoint(model_path))

            print("Testing model ...")
            eval_metrics = np.zeros(len(metric_tags))

            # Time the testing
            start_time = time.time()
            eval_metrics[0], output_probs = sess.run(output_feed,
                                                     input_feed)
            duration = time.time() - start_time
            eval_metrics[1:] += calculate_metrics(
                y, np.argmax(output_probs, axis=1))

            if self.args['results_file'] is None:
                if conf_dict is not None:
                    # Save the config to json
                    save_config_dict(conf_dict, model_path,
                                     metric_tags[:-1],
                                     list(eval_metrics[:-1]))
            else:
                # Create the file with headers if it doesn't exist
                if not os.path.isfile(self.args['results_file']):
                    with open(self.args['results_file'], 'w') as fd:
                        writer = csv.writer(fd)
                        writer.writerow(metric_tags[:-1])
                with open(self.args['results_file'],'a') as fd:
                    writer = csv.writer(fd)
                    row = list(eval_metrics[:-1])
                    writer.writerow(row)

            print()
            print(self.name + ":"
                  "  |  test duration: {}".format(duration))
            print_metrics(metric_tags[:-1], eval_metrics[:-1])
