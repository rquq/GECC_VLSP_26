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

    def test_tokenizer_identity_preserves_source_but_keeps_real_edits(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        from tools.screen_small_models import restore_tokenizer_identity

        backend = Tokenizer(models.WordLevel({'[UNK]': 0, 'this': 1, 'sentence': 2, 'correction': 3}, unk_token='[UNK]'))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]')
        source = 'this  sentence'
        self.assertEqual(restore_tokenizer_identity(tokenizer, source, 'this sentence'), source)
        self.assertEqual(restore_tokenizer_identity(tokenizer, source, 'this correction'), 'this correction')
        self.assertEqual(restore_tokenizer_identity(tokenizer, source, ''), '')

    def test_edit_supervision_covers_insertions_and_deletions(self):
        from tools.screen_small_models import edit_token_weights
        self.assertEqual(edit_token_weights([1, 2, 3], [1, 2, 3], 5), [1., 1., 1.])
        self.assertEqual(edit_token_weights([1, 3], [1, 2, 3], 5), [1., 5., 1.])
        self.assertEqual(edit_token_weights([1, 2, 3], [1, 3], 5), [1., 5.])

    def test_sequence_weighting_ignores_padding_and_emphasizes_error(self):
        import torch
        from tools.screen_small_models import weighted_sequence_loss
        logits = torch.tensor([[[4., 0.], [4., 0.], [0., 100.]]])
        labels = torch.tensor([[0, 1, -100]])
        weighted = weighted_sequence_loss(logits, labels, torch.tensor([[1., 5., 0.]]))
        unweighted = weighted_sequence_loss(logits, labels, torch.tensor([[1., 1., 0.]]))
        self.assertGreater(weighted.item(), unweighted.item())
        torch.testing.assert_close(weighted, weighted_sequence_loss(logits[:, :2], labels[:, :2],
                                                                    torch.tensor([[1., 5.]])))


if __name__ == '__main__':
    unittest.main()
