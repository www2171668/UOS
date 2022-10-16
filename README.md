# Hierarchical Reinforcement Learning with Unlimited Option Scheduling for Sparse Rewards in Continuous Space

## Tutorial
Our code is designed based on Agent Learning Framework (ALF), which  is a reinforcement learning framework.

## Algorithms
In hidio example package, there are two main algorithms, including hidio and uos.

## Installation
Installation and usage instructions can be found in ALF https://github.com/HorizonRobotics/alf.

alf/environments/Oly contains the environment of Tracks in our paper. If you want to examine methods on Tracks, you need to install requirements.txt and setup.py in Oly package.

## Examples
Our code is based on an older version of ALF used [gin](https://github.com/google/gin-config)
for job configuration. Its syntax is not as flexible as ALF conf (e.g., you can't easily
do math computation in a gin file). There are still some examples with `.gin`
under `alf/examples`. We are in the process of converting all `.gin` examples to `_conf.py`
examples.

You can train any `.gin` file under `alf/examples` using the following command:
```bash
cd alf/examples; python -m alf.bin.train --gin_file=GIN_FILE --root_dir=LOG_DIR
```
* GIN_FILE is the path to the gin conf (some `.gin` files under `alf/examples` might be invalid; they have not been converted to use the latest pytorch version of ALF).
* LOG_DIR has the same meaning as in the ALF conf example above.

## Acknowledge
Thanks to ALF and his team. Their code is very efficient and easy to use. They are also always quick to answer our questions and keep our work moving fast