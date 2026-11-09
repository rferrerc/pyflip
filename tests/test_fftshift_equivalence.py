
import os
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)

from propagation_classes import BroadbandWavefront, crop_to


CASES = (
    (32, 1, 1),
    (32, 2, 3),
    (64, 1, 3),
    (64, 2, 5),
    (31, 1, 1),
    (31, 2, 3),
    (33, 2, 1),
    (63, 2, 3),
)


def reference_fpm(model, field, mask, oversample):
    field = model.propagate(field, pad=oversample)
    field = field * mask
    field = torch.fft.fft2(
        torch.fft.fftshift(field, dim=(-2, -1))
    )
    return crop_to(field, model.npixels)


class TestFPMEquivalence(unittest.TestCase):

    def check_case(self, n, oversample, wavelengths):
        device = os.environ.get("WO004_TEST_DEVICE", "cpu")

        torch.manual_seed(20261108)

        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")

        padding = (
            n * (oversample - 1) // 2
            if oversample > 1 else 0
        )
        size = n + 2 * padding

        real = torch.randn(
            1, wavelengths, n, n,
            dtype=torch.float64, device=device
        )

        imaginary = torch.randn_like(real)
        initial = torch.complex(real, imaginary)

        initial_mask = torch.randn(
            1, wavelengths, size, size,
            dtype=torch.float32, device=device
        )

        x = initial.detach().clone().requires_grad_(True)
        y = initial.detach().clone().requires_grad_(True)

        a = initial_mask.detach().clone().requires_grad_(True)
        b = initial_mask.detach().clone().requires_grad_(True)

        model = BroadbandWavefront.__new__(
            BroadbandWavefront
        )
        torch.nn.Module.__init__(model)
        model.npixels = n

        expected = reference_fpm(model, x, a, oversample)
        actual = model.forward_fpm(y, b, oversample)

        torch.testing.assert_close(
            actual, expected, rtol=1e-7, atol=1e-8
        )

        expected.abs().square().mean().backward()
        actual.abs().square().mean().backward()

        torch.testing.assert_close(
            y.grad, x.grad, rtol=1e-7, atol=1e-8
        )
        torch.testing.assert_close(
            b.grad, a.grad, rtol=1e-7, atol=1e-8
        )


def make_test(n, oversample, wavelengths):
    def test(self):
        self.check_case(n, oversample, wavelengths)
    return test


for index, case in enumerate(CASES):
    setattr(
        TestFPMEquivalence,
        "test_configuration_%02d" % index,
        make_test(*case)
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
