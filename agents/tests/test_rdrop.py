"""R-Drop numerical, autograd and real Trainer accumulation contracts."""
import tempfile
import unittest
from types import SimpleNamespace

from model.classification import (_symmetric_joint_kl, _weighted_hierarchical_loss,
                                  _weighted_rdrop_loss, _rdrop_trainer_class)


class RDropTests(unittest.TestCase):
    def test_identical_shifted_and_extreme_distributions(self):
        import torch
        first = torch.tensor([[0.3, 1., -2., 3., 0.], [1000., -1000., 1000., 0., 500.]],
                             requires_grad=True)
        # A common subtype-logit shift leaves the joint distribution unchanged.
        same = first.detach().clone()
        same[:, 1:] += 8
        torch.testing.assert_close(_symmetric_joint_kl(first, same), torch.zeros(2), atol=1e-6, rtol=0)
        second = torch.tensor([[-1., -2., 1., 0., 3.], [-1000., 1000., -1000., 500., 0.]],
                              requires_grad=True)
        kl = _symmetric_joint_kl(first, second)
        self.assertEqual(kl.dtype, torch.float32)
        self.assertTrue((kl > 0).all().item())
        self.assertTrue(torch.isfinite(kl).all().item())
        for gradient in torch.autograd.grad(kl.sum(), (first, second)):
            self.assertTrue(torch.isfinite(gradient).all().item())
            self.assertGreater(gradient.abs().sum().item(), 0)
        torch.testing.assert_close(kl, _symmetric_joint_kl(second, first))

    def test_clean_subtypes_are_regularized_and_both_passes_get_gradients(self):
        import torch
        first = torch.tensor([[0.5, -2., 1., 3., 0.]], requires_grad=True)
        second = torch.tensor([[0.5, 2., 1., -3., 0.]], requires_grad=True)
        labels = torch.tensor([0])
        weights = torch.tensor([0.7, 1., 2., 3., 4.])
        supervised = 0.5 * (_weighted_hierarchical_loss(first, labels, weights)
                            + _weighted_hierarchical_loss(second, labels, weights))
        for gradient in torch.autograd.grad(supervised, (first, second), retain_graph=True):
            torch.testing.assert_close(gradient[:, 1:], torch.zeros(1, 4))
        loss = _weighted_rdrop_loss(first, second, labels, weights)
        torch.testing.assert_close(loss - supervised, 0.7 * _symmetric_joint_kl(first, second).sum())
        for gradient in torch.autograd.grad(loss, (first, second)):
            self.assertGreater(gradient[:, 1:].abs().sum().item(), 0)

    def test_full_and_partial_window_loss_and_gradient_equivalence(self):
        import torch
        for rows in (3, 8):
            torch.manual_seed(42)
            first = torch.randn(rows, 5, requires_grad=True)
            second = torch.randn(rows, 5, requires_grad=True)
            labels = torch.arange(rows) % 5
            weights = torch.tensor([0.5, 1., 1.5, 2., 2.5])
            full = _weighted_rdrop_loss(first, second, labels, weights)
            expected = torch.autograd.grad(full, (first, second), retain_graph=True)
            split = sum(_weighted_rdrop_loss(first[i:i+1], second[i:i+1], labels[i:i+1],
                                             weights, rows) for i in range(rows))
            torch.testing.assert_close(split, full)
            for actual, wanted in zip(torch.autograd.grad(split, (first, second)), expected):
                torch.testing.assert_close(actual, wanted)

    def test_trainer_independent_dropout_label_removal_first_output_and_eval(self):
        import torch
        from transformers import TrainingArguments

        class DropoutModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = torch.nn.Dropout(0.5)
                self.weight = torch.nn.Parameter(torch.ones(32, 5))
                self.outputs = []

            def forward(self, input_ids):
                output = SimpleNamespace(logits=self.dropout(input_ids.float()) @ self.weight)
                output.logits.retain_grad()
                self.outputs.append(output)
                return output

        with tempfile.TemporaryDirectory() as directory:
            model = DropoutModel()
            trainer = _rdrop_trainer_class()(model=model, loss_weights=torch.ones(5),
                args=TrainingArguments(output_dir=directory, use_cpu=True, report_to="none"))
            inputs = {"input_ids": torch.ones(4, 32), "labels": torch.arange(4)}
            torch.manual_seed(42)
            loss, output = trainer.compute_loss(model, inputs, True, num_items_in_batch=8)
            self.assertIs(output, model.outputs[0])
            self.assertIn("labels", inputs)
            self.assertEqual(len(model.outputs), 2)
            self.assertFalse(torch.equal(model.outputs[0].logits, model.outputs[1].logits))
            loss.backward()
            for result in model.outputs:
                self.assertGreater(result.logits.grad.abs().sum().item(), 0)
            model.eval()
            trainer.compute_loss(model, inputs)
            self.assertEqual(len(model.outputs), 3)

    def test_real_trainer_does_not_divide_window_loss_twice(self):
        import torch
        from transformers import TrainingArguments

        class LinearModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.arange(15).reshape(3, 5).float() / 20)

            def forward(self, input_ids):
                return {"logits": input_ids.float() @ self.weight}

        # Return attribute-style logits for compute_loss, with no native labels/loss.
        class Model(LinearModel):
            def forward(self, input_ids):
                return SimpleNamespace(**super().forward(input_ids))

        for rows in (3, 8):
            data = [{"input_ids": [1., float(i + 1), -0.5], "labels": i % 5} for i in range(rows)]
            weights = torch.tensor([0.5, 1., 1.5, 2., 2.5])
            expected = Model()
            logits = expected(torch.tensor([row["input_ids"] for row in data])).logits
            loss = _weighted_rdrop_loss(logits, logits, torch.tensor([r["labels"] for r in data]), weights)
            loss.backward()
            wanted = expected.weight.detach() - 0.03 * expected.weight.grad
            with tempfile.TemporaryDirectory() as directory:
                model = Model()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
                args = TrainingArguments(output_dir=directory, use_cpu=True, report_to="none",
                    per_device_train_batch_size=1, gradient_accumulation_steps=8, max_steps=1,
                    label_names=["labels"], max_grad_norm=0, save_strategy="no",
                    logging_strategy="no", disable_tqdm=True)
                trainer = _rdrop_trainer_class()(model=model, args=args, train_dataset=data,
                    loss_weights=weights, optimizers=(optimizer, None))
                trainer.train()
                torch.testing.assert_close(model.weight, wanted, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
