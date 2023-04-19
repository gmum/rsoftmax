import unittest
import torch
from rsoftmax import RSoftmax
from utils import is_valid_probability_distribution


class TestRSoftmax(unittest.TestCase):
    def test_simple_case(self):
        r_softmax = RSoftmax(dim=-1, eps=1e-8)
        input_tensor = torch.tensor([[3, 2, 1],
                                     [4, 5, 6]], dtype=torch.float32)

        expected_output = torch.tensor([[1, 0, 0],
                                        [0, 0, 1]], dtype=torch.float32)
        r = torch.tensor([0.5], dtype=torch.float32)
        output_tensor = r_softmax(input_tensor, r)
        self.assertTrue(is_valid_probability_distribution(output_tensor, dim=-1))
        self.assertTrue(torch.allclose(output_tensor, expected_output, rtol=1e-3))

    def test_simple_case2(self):
        r_softmax = RSoftmax(dim=-1, eps=1e-8)
        input_tensor = torch.tensor([[3, 2, 1, 1],
                                     [4, 6, 6, 5]], dtype=torch.float32)

        expected_output = torch.tensor([[0.8908, 0.1092, 0, 0],
                                        [0, 0.5, 0.5, 0]], dtype=torch.float32)
        r = torch.tensor([0.5], dtype=torch.float32)
        output_tensor = r_softmax(input_tensor, r)
        self.assertTrue(is_valid_probability_distribution(output_tensor, dim=-1))
        self.assertTrue(torch.allclose(output_tensor, expected_output, rtol=1e-3))

    def test_simple_case3(self):
        r_softmax = RSoftmax(dim=-1, eps=1e-8)
        input_tensor = torch.tensor([[3, 2, 1, 1],
                                     [4, 6, 6, 5]], dtype=torch.float32)

        expected_output = torch.tensor([[1.0, 0, 0, 0],
                                        [0, 0.5, 0.5, 0]], dtype=torch.float32)
        r = torch.tensor([0.75], dtype=torch.float32)
        output_tensor = r_softmax(input_tensor, r)
        self.assertTrue(is_valid_probability_distribution(output_tensor, dim=-1))
        self.assertTrue(torch.allclose(output_tensor, expected_output, rtol=1e-3))

    def test_different_r(self):
        r_softmax = RSoftmax(dim=-1, eps=1e-8)
        input_tensor = torch.tensor([[3, 2, 1, 1],
                                     [4, 6, 5, 5]], dtype=torch.float32)

        expected_output = torch.tensor([[1.0, 0, 0, 0],
                                        [0, 0.8717, 0.0641, 0.0641]], dtype=torch.float32)
        r = torch.tensor([[0.75], [0.25]], dtype=torch.float32)
        output_tensor = r_softmax(input_tensor, r)
        self.assertTrue(is_valid_probability_distribution(output_tensor, dim=-1))
        self.assertTrue(torch.allclose(output_tensor, expected_output, rtol=1e-3))

    def test_zeros_input(self):
        r_softmax = RSoftmax(dim=-1, eps=1e-8)
        input_tensor = torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.float32)
        expected_output = torch.tensor([[0.33333334, 0.33333334, 0.33333334],
                                        [0.33333334, 0.33333334, 0.33333334]], dtype=torch.float32)
        r = torch.tensor([0.5], dtype=torch.float32)
        output_tensor = r_softmax(input_tensor, r)
        self.assertTrue(is_valid_probability_distribution(output_tensor, dim=-1))
        self.assertTrue(torch.allclose(output_tensor, expected_output, rtol=1e-3))

    def test_negative_input(self):
        r_softmax = RSoftmax(dim=-1, eps=1e-8)
        input_tensor = torch.tensor([[-2.4, -4.3, -1.3, -34.3], [-20.3, -2.3, -42.3, -1e-6]], dtype=torch.float32)
        expected_output = torch.tensor([[0, 0, 1, 0],
                                        [0, 0, 0, 1]], dtype=torch.float32)
        r = torch.tensor([0.75], dtype=torch.float32)
        output_tensor = r_softmax(input_tensor, r)
        self.assertTrue(is_valid_probability_distribution(output_tensor, dim=-1))
        self.assertTrue(torch.allclose(output_tensor, expected_output, rtol=1e-3))


if __name__ == '__main__':
    unittest.main()
