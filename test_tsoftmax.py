import unittest
import torch
from torch import tensor
from rsoftmax import TSoftmax
from utils import is_valid_probability_distribution


class TestTSoftmax(unittest.TestCase):
    def test_simple_case(self):
        t_softmax = TSoftmax(dim=-1)
        input_tensor = tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        t = tensor([0.5], dtype=torch.float32)
        output_tensor = t_softmax(input_tensor, t)
        self.assertTrue(is_valid_probability_distribution(output_tensor, dim=-1))

    def test_negative_input(self):
        t_softmax = TSoftmax(dim=-1, eps=1e-8)
        input_tensor = tensor([[-1, -2, -3], [-4, -5, -6]], dtype=torch.float32)
        t = tensor([0.5], dtype=torch.float32)
        output_tensor = t_softmax(input_tensor, t)
        self.assertTrue(is_valid_probability_distribution(output_tensor, dim=-1))


if __name__ == '__main__':
    unittest.main()