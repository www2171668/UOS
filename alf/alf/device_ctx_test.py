

import torch
import alf


class DeviceCtxTest(alf.test.TestCase):
    def test_device_ctx(self):
        with alf.device("cpu"):
            self.assertEqual(alf.get_default_device(), "cpu")
            self.assertEqual(torch.tensor([1]).device.type, "cpu")
            if torch.cuda.is_available():
                with alf.device("cuda"):
                    self.assertEqual(alf.get_default_device(), "cuda")
                    self.assertEqual(torch.tensor([1]).device.type, "cuda")
            self.assertEqual(alf.get_default_device(), "cpu")
            self.assertEqual(torch.tensor([1]).device.type, "cpu")


if __name__ == "__main__":
    alf.test.main()
