"""Hierarchical Trainer full and partial accumulation-window update contracts."""
import tempfile
import unittest
from types import SimpleNamespace

from model.classification import _weighted_hierarchical_loss, _hierarchical_trainer_class


class HierarchicalTrainerTests(unittest.TestCase):
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
            loss = _weighted_hierarchical_loss(logits, torch.tensor([r["labels"] for r in data]), weights)
            loss.backward()
            wanted = expected.weight.detach() - 0.03 * expected.weight.grad
            with tempfile.TemporaryDirectory() as directory:
                model = Model()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
                args = TrainingArguments(output_dir=directory, use_cpu=True, report_to="none",
                    per_device_train_batch_size=1, gradient_accumulation_steps=8, max_steps=1,
                    label_names=["labels"], max_grad_norm=0, save_strategy="no",
                    logging_strategy="no", disable_tqdm=True)
                trainer = _hierarchical_trainer_class()(model=model, args=args, train_dataset=data,
                    loss_weights=weights, optimizers=(optimizer, None))
                trainer.train()
                torch.testing.assert_close(model.weight, wanted, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
