import random
import copy
import json

import numpy as np
from sklearn import metrics


def get_validset_feeds(model, x, y, h_dim):
    x_new, v_seq_lengths = pad_seqs(x)

    input_feed = {model.x: x_new,
                  model.y: y,
                  model.seq_lens: v_seq_lengths,
                  model.training: False,
                  model.keep_prob: 1.0,
                  }

    output_feed = [model.loss_nowd, model.output_probs]

    return input_feed, output_feed


def save_config_dict(config, path, tags=None, values=None):
    '''
    Save a dictionary (config) to a json file in path

    Input:
    config  - A dictionary to save
    path    - String specifying the path
    tags    - (optional) The keys to add to the dict
    values  - (optional) The values to add to the dict
    '''
    if '.json' not in path:
        path = path+'config.json'

    if tags is not None and values is not None:
        for tag, val in zip(tags, values):
            if type(val) is np.ndarray:
                config[tag] = list(val)
            else:
                config[tag] = val
                if type(val) == np.float32:
                    config[tag] = val.astype(np.float64)

    with open(path, 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)


def create_conf_dict(args):
    '''
    Create the config dictionary from tensorflow flags.
    '''
    d = copy.deepcopy(vars(args))  # Important to copy the dict,
                                   # otherwise the original args
                                   # dictionary will be mutated.
    del d['logdir']
    del d['test']
    del d['gpu']

    return d


def calculate_metrics(y_true, y_pred, labels=None):
    '''
    A function to compute the accuracy, F1 score.
    The 'macro' F1 score is computed; meaning the F1 score is computed for each
    class and then the unwieghted average is returned.

    Input:
    y_true - a numpy array of true labels.
    y_pred - a numpy array of predicted labels.
    labels - the labels for each class in y_true with shape (max(y_true)+1).

    Returns:
    A numpy array of Accuracy and F1
    '''
    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true, axis=1)
    elif len(y_true.shape) == 3:
        y_flat = np.reshape(y_true, (-1, y_true.shape[-1]))
        y_true = np.argmax(y_flat, axis=1)

    if len(y_pred.shape) == 2:
        y_pred = np.argmax(y_pred, axis=1)
    elif len(y_pred.shape) == 3:
        y_flat = np.reshape(y_pred, (-1, y_pred.shape[-1]))
        y_pred = np.argmax(y_flat, axis=1)

    acc = np.average(np.equal(y_true, y_pred))

    F1 = metrics.f1_score(y_true, y_pred, labels, average='macro')

    return np.array([acc, F1])


def get_max_length(var):
    '''
    A function that returns the largest length of an array in a list of arrays.

    Input:
    var - A list of numpy arrays

    Returns:
    The largest length (int)
    '''
    return len(max(var, key=lambda v: len(v)))


def print_metrics(tags, values):
    '''
    Prints the tags and values column-wise
    The columns have a width of 13 characters and tags should not be
    longer than 10 characters
    '''
    label_str = ""
    value_str = ''
    for i in range(len(tags)):
        label_str += tags[i].rjust(13)
        value_str += '{:13.4}'.format(values[i])

    print(label_str)
    print(value_str)


def get_minibatches_indices(n, minibatch_size, shuffle=True,
                            shuffleBatches=True,
                            allow_smaller_final_batch=False):
    """
    Inputs:
    n - An integer corresponding to the total number of data points

    Returns:
    minibatches - a list of lists with indices in the lowest dimension
    """
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    if n <= minibatch_size:
        return [idx_list]

    minibatches = []
    minibatch_start = 0

    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size
    if allow_smaller_final_batch:
        if (minibatch_start != n):
            # Make a minibatch out of what is left
            minibatches.append(idx_list[minibatch_start:])

    if shuffleBatches is True:
        # minibatches here is a list of list indexes
        random.shuffle(minibatches)

    return minibatches


def random_split(n, test_frac=0.1):
    all_idx = np.arange(n)
    test_idx = all_idx[np.random.choice(
        n, int(np.ceil(test_frac * n)), replace=False)]
    train_idx = np.setdiff1d(all_idx, test_idx)
    assert (np.all(np.sort(np.hstack([train_idx, test_idx])) == all_idx))
    return train_idx, test_idx


def randomly_split_data(y, test_frac=0.5, valid_frac=0):
    '''
    Split the data into 3 sets:

    Returns a tuple with:
        train-idx       - A list of the train indices.
        valid-idx       - A list of the validation indices.
        test-idx        - A list of the testing indices.
    '''
    if len(y.shape) == 1:
        print('Warning: are you sure Y contains all the classes?')
        y = one_hot_encoder(y)

    split = None
    smallest_class = min(y.sum(axis=0))

    while split is None:
        not_test_idx, test_idx = random_split(
            y.shape[0], test_frac=test_frac + valid_frac)

        cond1 = np.all(y[not_test_idx, :].sum(axis=0) >=
                       0.8 * (1 - test_frac - valid_frac) * smallest_class)
        cond2 = np.all(y[test_idx, :].sum(axis=0) >=
                       0.8 * (test_frac + valid_frac) * smallest_class)

        if cond1 and cond2:
            if valid_frac != 0:
                while split is None:
                    final_test_idx, valid_idx = random_split(
                        y[test_idx].shape[0],
                        test_frac=valid_frac / (test_frac + valid_frac))

                    cond1 = np.all(
                        y[test_idx, :][final_test_idx, :].sum(axis=0) >=
                        0.6 * test_frac * smallest_class)
                    cond2 = np.all(
                        y[test_idx, :][valid_idx, :].sum(axis=0) >=
                        0.6 * valid_frac * smallest_class)
                    if cond1 and cond2:
                        split = (np.sort(not_test_idx),
                                 np.sort(test_idx[valid_idx]),
                                 np.sort(test_idx[final_test_idx]))
                        print('Split completed.\n')
                        break
                    else:
                        print('Valid labels unevenly split, resplitting...\n')
                        print(y[test_idx, :][final_test_idx, :].sum(axis=0))
                        print(0.6 * test_frac * smallest_class)
                        print(y[test_idx, :][valid_idx, :].sum(axis=0))
                        print(0.6 * valid_frac * smallest_class)
            else:
                split = (np.sort(not_test_idx), None, np.sort(test_idx))
                print('Split completed.\n')
        else:
            print('Test labels unevenly split, resplitting...\n')
            print(y[not_test_idx, :].sum(axis=0))
            print(0.7 * (1 - test_frac) * smallest_class)
            print(y[test_idx, :].sum(axis=0))
            print(0.7 * test_frac * smallest_class)

    if valid_frac != 0:
        print("Train:Validation:Testing - %d:%d:%d" % (len(split[0]),
                                                       len(split[1]),
                                                       len(split[2])))
    else:
        print("Train:Testing - %d:%d" % (len(split[0]), len(split[2])))

    return split


def one_hot_sequence(x, depth=None):
    if depth is None:
        depth = int(x.max() + 1)

    one_hot = np.zeros(x.shape + (depth,))
    # TODO: speed up this operation its super slow.
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            one_hot[i,j,int(x[i,j])] = 1

    return one_hot


def one_hot_encoder(labels, n_classes=None):
    '''
    A function that converts an array of integers, representing the labels
    of data, into an array of one-hot encoded vectors.

    Inputs:
    Labels      - a numpy array of integers (nsamples)
    n_classes   - (int) the number of classes if labels does not contain all

    Returns:
    A numpy array of one-hot encoded vectors with size [nsamples, n_classes]
    '''
    if n_classes is None:
        n_classes = int(labels.max() + 1)

    one_hot = np.zeros((labels.shape[0], n_classes))
    one_hot[np.arange(labels.shape[0]), labels.astype(int)] = 1
    return one_hot


def pad_seqs(seqs):
    """Create the matrices from the datasets.

    This pad each sequence to the same length: the length of the
    longest sequence.

    Output:
    x -- a numpy array with shape (batch_size, max_time_steps, num_features)
    lengths -- an array of the sequence lengths

    """
    lengths = [len(s) for s in seqs]

    n_samples = len(seqs)
    maxlen = np.max(lengths)
    inputDimSize = seqs[0].shape[1]

    x = np.zeros((n_samples, maxlen, inputDimSize))

    for idx, s in enumerate(seqs):
        x[idx, :lengths[idx], :] = s

    return x, np.array(lengths)
