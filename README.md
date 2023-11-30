# Continuous-Time Neural CDEs
Code release for our paper here: https://arxiv.org/abs/2311.04986

## Installation

```conda env create -f environment.yaml```

## Dataset setup

For Moving MNIST (https://www.cs.toronto.edu/~nitish/unsupervised_video/):

```cd datasets/MNIST && chmod +x download.sh && ./download.sh```

For Kinetics (https://www.crcv.ucf.edu/data/UCF101.php):

```cd datasets/UCF101 && chmod +x download.sh && ./download.sh```

## Running

To start a training run, run the following (leave the comma behind the `0,`):

```PYTHONPATH=. python main.py --base <path_to_config> --devices 0,```