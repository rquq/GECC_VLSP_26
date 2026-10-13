import unittest


class WeightedScreenLossTests(unittest.TestCase):
    def test_microbatch_accumulation_preserves_class_weighting(self):
        import torch
        from tools.screen_small_models import weighted_classification_loss

        logits = torch.tensor([[2., 0., 0.], [0., 2., 0.]], requires_grad=True)
        labels, weights = torch.tensor([0, 1]), torch.tensor([1., 3., 1.])
        full = weighted_classification_loss(logits, labels, weights)
        micro = sum(weighted_classification_loss(logits[i:i+1], labels[i:i+1], weights)
                    for i in range(2)) / 2
        torch.testing.assert_close(full, micro)
        gradient, = torch.autograd.grad(micro, logits)
        self.assertAlmostEqual(abs(gradient[1, 1].item() / gradient[0, 0].item()), 3.)


if __name__ == '__main__':
    unittest.main()
