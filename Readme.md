# Janet
Code for the forget-gate only LSTM in the paper ["The unreasonable effectiveness of the forget gate"](https://arxiv.org/abs/1804.04849)

Python 3.5

Tensorflow 1.4

### Gettting started

Install Tensorflow 1.4

Run:
```bash
pip install -r requirements.txt
```
Add and Copy experiments can be run with

```bash
python main.py --data add --name my_add_exp --log_every 20 --batch_size 50 --epochs 30
```
```bash
python main.py --data copy --name my_copy_exp --log_every 20 --batch_size 50 --epochs 30
```	

View the results on Tensorboard with
```bash
Tensorboard --logdir log
```

Use the ```--cell``` argument to set the type of cell, default is janet. E.g.,
```bash
python main.py --data copy --name my_copy_exp --log_every 20 --batch_size 50 --epochs 30 --cell lstm
```

Use the ```--chrono``` argument to use chrono initialization for the LSTM. E.g.,
```bash
python main.py --data copy --name my_copy_exp --log_every 20 --batch_size 50 --epochs 30 --cell lstm --chrono
```

MNIST and pMNIST experiments can be run with
```bash
python main.py --data mnist --name my_mnist_exp  --layers 128,128 --wd 0.00001 --epochs 100
```
```bash
python main.py --data pmnist --name my_pmnist_exp  --wd 0.00001 --epochs 100
```	

Use the *_exp.py files to run multiple experiments. E.g.,
```bash
python other_exp.py --data pmnist --name multi_pmnist --wd 0.00001 --cell lstm
```

### Forget-only cell
The changes to the LSTM cell are in aux_code/rnn_cells.py
