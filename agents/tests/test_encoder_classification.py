"""Full/partial accumulation must preserve encoder sample weighting."""
import unittest

from model.encoder_classification import weighted_loss


class EncoderLossTests(unittest.TestCase):
    def test_weighted_full_and_partial_windows_preserve_gradients(self):
        import torch
        import torch.nn.functional as F

        for rows in (3, 8):
            torch.manual_seed(42)
            logits = torch.randn(rows, 5, requires_grad=True)
            labels = torch.arange(rows) % 5
            weights = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5])
            expected = (F.cross_entropy(logits, labels, reduction="none") * weights[labels]).sum() / rows
            full = weighted_loss(logits, labels, weights)
            split = sum(weighted_loss(logits[i:i + 1], labels[i:i + 1], weights, rows)
                        for i in range(rows))
            torch.testing.assert_close(full, expected)
            torch.testing.assert_close(split, full)
            wanted = torch.autograd.grad(expected, logits, retain_graph=True)[0]
            torch.testing.assert_close(torch.autograd.grad(split, logits)[0], wanted)


if __name__ == "__main__":
    unittest.main()
