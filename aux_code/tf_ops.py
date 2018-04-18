import tensorflow as tf


def create_sess():
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    return tf.Session(config=sess_config)


def create_scalar_summaries(tags, values):
    '''
    Input:
    tags - the tag names to use in the summary
    values - the values to log

    Returns:
    A tensorflow summary to be used in wrtier.add_summary(summary, steps)
    '''
    summary_value_list = []
    for i in range(len(tags)):
        summary_value_list.append(tf.Summary.Value(tag=tags[i],
                                                   simple_value=values[i]))
    return tf.Summary(value=summary_value_list)


def linear(input_data, output_dim, scope=None, stddev=1.0, init_func=None):
    if init_func == 'norm':
        initializer = tf.random_normal_initializer(stddev=stddev)
    elif init_func is None:
        initializer = None
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable(
            'weights', [input_data.get_shape()[-1], output_dim],
            initializer=initializer)
        b = tf.get_variable('bias', [output_dim], initializer=const)
        return tf.matmul(input_data, w) + b
