import os
# Turn off the tensorflow warnings about code not being compiled with some
# optimizations. (switch off warning logging, but not error logging)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # noqa

import tensorflow as tf

import main

args = main.parse_args()

# Set the default args
config_dict = {}
config_dict['epochs'] = 100

args_dict = vars(args)
name = args.name

# Loop over the number of independent runs
for r in range(10):
    print('*'*50)
    print('Run number {}'.format(r))

    config_dict['results_file'] = 'experiments/' + name + '.csv'
    args.name = name + '_g' + str(r+1)

    args_dict.update(config_dict)
    main.main(args)
    tf.reset_default_graph()
