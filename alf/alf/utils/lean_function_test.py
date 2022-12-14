

import copy
import torch

import alf
from alf.utils.lean_function import lean_function


def func(x, w, b, scale=1.0):
    return torch.sigmoid(scale * (x @ w) + b)


class TestLeanFunction(alf.test.TestCase):
    def test_lean_function(self):
        x = torch.randn((3, 4), requires_grad=True)
        w = torch.randn((4, 5), requires_grad=True)
        b = torch.randn(5, requires_grad=True)
        lean_func = lean_function(func)
        y1 = func(x, w, b)
        y2 = lean_func(x, w, b)
        self.assertTensorEqual(y1, y2)
        grad1 = torch.autograd.grad(y1.sum(), x)[0]
        grad2 = torch.autograd.grad(y2.sum(), x)[0]
        self.assertTensorEqual(grad1, grad2)

        y3 = lean_func(x, b=b, w=w)
        self.assertTensorEqual(y1, y3)
        grad3 = torch.autograd.grad(y3.sum(), x)[0]
        self.assertTensorEqual(grad1, grad3)

        y1 = func(x, w, b, scale=2.0)
        y2 = lean_func(x, w=w, b=b, scale=2.0)
        self.assertTensorEqual(y1, y2)
        grad1 = torch.autograd.grad(y1.sum(), x)[0]
        grad2 = torch.autograd.grad(y2.sum(), x)[0]
        self.assertTensorEqual(grad1, grad2)

    def test_lean_function_module(self):
        func1 = alf.layers.FC(3, 5, activation=torch.relu_)
        func2 = copy.deepcopy(func1)
        x = torch.randn((4, 3), requires_grad=True)
        func2 = lean_function(func2)
        y1 = func1(x)
        y2 = func2(x)
        self.assertTensorEqual(y1, y2)

        grad1 = torch.autograd.grad(y1.sum(), x)[0]
        grad2 = torch.autograd.grad(y2.sum(), x)[0]
        self.assertTensorEqual(grad1, grad2)

        y1 = func1(x)
        y2 = func2(x)
        y1.sum().backward()
        y2.sum().backward()
        for p1, p2 in zip(func1.parameters(), func2.parameters()):
            self.assertTensorEqual(p1.grad, p2.grad)

    def test_lean_function_network(self):
        func1 = alf.nn.Sequential(
            alf.layers.FC(3, 5, activation=torch.relu_),
            alf.layers.FC(5, 1, activation=torch.sigmoid))
        func2 = func1.copy()
        for p1, p2 in zip(func1.parameters(), func2.parameters()):
            p2.data.copy_(p1)
        x = torch.randn((4, 3), requires_grad=True)
        func2 = lean_function(func2)
        y1 = func1(x)[0]
        y2 = func2(x)[0]
        self.assertTensorEqual(y1, y2)

        grad1 = torch.autograd.grad(y1.sum(), x)[0]
        grad2 = torch.autograd.grad(y2.sum(), x)[0]
        self.assertTensorEqual(grad1, grad2)

        y1 = func1(x)[0]
        y2 = func2(x)[0]
        y1.sum().backward()
        y2.sum().backward()
        for p1, p2 in zip(func1.parameters(), func2.parameters()):
            self.assertTensorEqual(p1.grad, p2.grad)


if __name__ == '__main__':
    alf.test.main()
