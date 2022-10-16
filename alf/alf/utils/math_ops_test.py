

import torch

import alf


class MathOpsTest(alf.test.TestCase):
    def test_argmin(self):
        a = torch.tensor([[2, 5, 2], [0, 1, 2], [1, 1, 1], [3, 2, 1],
                          [4, 2, 2]])
        i = alf.math.argmin(a)
        self.assertEqual(i, torch.tensor([0, 0, 0, 2, 1]))

    def test_softsign(self):
        input = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(alf.math.softsign, input)
        # gradcheck cannot handle inplace op directly, so we need to add another
        # op before softsign_
        torch.autograd.gradcheck(lambda x: alf.math.softsign_(x * 1.0), input)


if __name__ == '__main__':
    alf.test.main()
