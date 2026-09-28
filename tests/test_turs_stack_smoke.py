import torch


def test_turs_stack_model_smoke_import_and_forward():
    from models.turs_stack.model import TURSStack

    model = TURSStack(in_channels=1, num_classes=5, sequence_length=140)
    x = torch.randn(2, 1, 140)
    out = model(x)

    assert isinstance(out, dict)
    assert "probs" in out
    assert "branch_logits" in out
    assert len(out["probs"]) == 4
    for p in out["probs"]:
        assert p.shape[0] == 2
        assert p.shape[1] == 5
