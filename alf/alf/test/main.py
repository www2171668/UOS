

from absl import logging
import torch
import unittest


def main():
    logging.use_absl_handler()
    logging.set_verbosity(logging.INFO)

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    unittest.main()
