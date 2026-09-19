import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


MODULE_PATH = Path(__file__).with_name("train_joint_text_recurrent.py")
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location(
    "train_joint_text_recurrent", MODULE_PATH
)
joint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(joint)


class FakeTextModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = torch.nn.Module()
        self.bert.encoder = torch.nn.Module()
        self.bert.encoder.layer = torch.nn.ModuleList(
            [torch.nn.Linear(2, 2) for _ in range(4)]
        )
        self.embeddings = torch.nn.Linear(2, 2)
        self.classifier = torch.nn.Linear(2, 2)


class JointTextRecurrentTests(unittest.TestCase):
    def test_text_model_loader_forces_eager_attention(self):
        class Loader:
            called_with = None

            @classmethod
            def from_pretrained(cls, path, **kwargs):
                cls.called_with = (path, kwargs)
                return "model"

        model = joint.load_text_model("checkpoint", Loader)

        self.assertEqual(model, "model")
        self.assertEqual(Loader.called_with[0], "checkpoint")
        self.assertEqual(
            Loader.called_with[1]["attn_implementation"], "eager"
        )

    def test_only_requested_final_text_layers_are_trainable(self):
        model = FakeTextModel()

        parameters = joint.set_trainable_text_layers(model, final_layers=2)

        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.embeddings.parameters())
        )
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.classifier.parameters())
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for layer in model.bert.encoder.layer[:2]
                for parameter in layer.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for layer in model.bert.encoder.layer[-2:]
                for parameter in layer.parameters()
            )
        )
        self.assertEqual(
            {id(parameter) for parameter in parameters},
            {
                id(parameter)
                for layer in model.bert.encoder.layer[-2:]
                for parameter in layer.parameters()
            },
        )

    def test_freezing_text_returns_no_trainable_text_parameters(self):
        model = FakeTextModel()

        parameters = joint.set_trainable_text_layers(model, final_layers=0)

        self.assertEqual(parameters, [])
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

    def test_pool_current_tokens_uses_only_marked_tokens(self):
        hidden = torch.tensor(
            [[[1.0, 1.0], [3.0, 5.0], [7.0, 9.0], [20.0, 20.0]]]
        )
        mask = torch.tensor([[False, True, True, False]])

        pooled = joint.pool_current_tokens(hidden, mask)

        torch.testing.assert_close(pooled, torch.tensor([[5.0, 7.0]]))

    def test_build_optimizer_uses_separate_learning_rates(self):
        fusion = torch.nn.Linear(3, 2)
        text = torch.nn.Linear(3, 2)
        args = SimpleNamespace(
            learning_rate=2e-4,
            text_learning_rate=1e-5,
            weight_decay=0.01,
        )

        optimizer = joint.build_optimizer(
            fusion.parameters(), text.parameters(), args
        )

        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [2e-4, 1e-5],
        )


if __name__ == "__main__":
    unittest.main()
