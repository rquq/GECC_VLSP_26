"""Separate hierarchy weights, masked gradients and accumulation invariants."""
import unittest

from model.classification import _loss_weighting, _weighted_hierarchical_loss


class SeparateWeightTests(unittest.TestCase):
    def test_fit_local_normalization_and_binary_subtype_independence(self):
        def weights(counts):
            return _loss_weighting([{"label": label} for label, count in enumerate(counts)
                                    for _ in range(count)], 0.5, "separate")
        first = weights([20, 12, 2, 3, 3])
        second = weights([20, 2, 12, 3, 3])
        self.assertEqual(first["detection_weights"], second["detection_weights"])
        self.assertNotEqual(first["subtype_weights"], second["subtype_weights"])
        for counts, values in ((first["detection_counts"], first["detection_weights"]),
                               (first["subtype_counts"], first["subtype_weights"][1:])):
            self.assertAlmostEqual(sum(n * w for n, w in zip(counts, values)) / sum(counts), 1)
        self.assertGreater(first["subtype_weights"][2], first["subtype_weights"][1])
        unit = _loss_weighting([{"label": k} for k in range(5)], 0, "separate")
        self.assertEqual(unit["detection_weights"], [1, 1])
        self.assertEqual(unit["subtype_weights"], [0, 1, 1, 1, 1])

    def test_manual_objective_and_clean_gradient_mask(self):
        import torch
        import torch.nn.functional as F
        logits = torch.tensor([[0.2, 2., -1., 0., 1.], [-0.4, 0., 1., 2., -1.]], requires_grad=True)
        labels = torch.tensor([0, 3])
        shared = torch.tensor([0.5, 1., 1.5, 2., 2.5])
        binary = torch.tensor([0.8, 1.2])
        subtype = torch.tensor([0., 0.6, 1.7, 1.4, 1.3])
        actual = _weighted_hierarchical_loss(logits, labels, shared, 8, binary, subtype)
        expected = (0.8 * F.softplus(logits[0, 0]) + 1.2 * F.softplus(-logits[1, 0])
                    - 2 * 1.4 * F.log_softmax(logits[1, 1:], dim=-1)[2]) / 8
        torch.testing.assert_close(actual, expected)
        gradient = torch.autograd.grad(actual, logits)[0]
        torch.testing.assert_close(gradient[0, 1:], torch.zeros(4))
        self.assertGreater(gradient[1, 1:].abs().sum().item(), 0)

    def test_full_and_partial_window_loss_and_gradient_equivalence(self):
        import torch
        for rows in (3, 8):
            torch.manual_seed(42)
            first = torch.randn(rows, 5, requires_grad=True)
            labels = torch.arange(rows) % 5
            shared = torch.tensor([0.5, 1., 1.5, 2., 2.5])
            binary = torch.tensor([0.8, 1.2])
            subtype = torch.tensor([0., 0.6, 1.7, 1.4, 1.3])
            full = _weighted_hierarchical_loss(first, labels, shared, None, binary, subtype)
            gradients = torch.autograd.grad(full, first, retain_graph=True)[0]
            split = sum(_weighted_hierarchical_loss(first[i:i+1], labels[i:i+1],
                                                    shared, rows, binary, subtype) for i in range(rows))
            torch.testing.assert_close(split, full)
            torch.testing.assert_close(torch.autograd.grad(split, first)[0], gradients)


if __name__ == "__main__":
    unittest.main()
