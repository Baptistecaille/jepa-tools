import sys
import textwrap

import pytest
import torch

from jepa_tools.core import registry
from jepa_tools.core.registry import RegistryError, get, names, register
from jepa_tools.core.types import Plan, StepOutputs


@pytest.fixture(autouse=True)
def isolated_registry():
    """Snapshot the registry so test pieces never leak into other tests."""
    snapshot = {k: dict(v) for k, v in registry._REGISTRY.items()}
    yield
    for kind in registry._REGISTRY:
        registry._REGISTRY[kind].clear()
        registry._REGISTRY[kind].update(snapshot[kind])


def test_register_and_get():
    @register("encoder", "dummy")
    class Dummy:
        pass

    assert get("encoder", "dummy") is Dummy
    assert "dummy" in names("encoder")


def test_unknown_name_lists_available():
    @register("encoder", "vit")
    class Vit:
        pass

    with pytest.raises(RegistryError, match="vit") as exc:
        get("encoder", "vitt")
    assert "vitt" in str(exc.value)


def test_duplicate_name_rejected():
    @register("encoder", "dup")
    class A:
        pass

    with pytest.raises(RegistryError, match="dup"):
        @register("encoder", "dup")
        class B:
            pass


def test_unknown_kind_rejected():
    with pytest.raises(RegistryError, match="sampler"):
        register("encodr", "x")
    with pytest.raises(RegistryError, match="encoder"):
        names("encodr")


def test_registry_error_is_value_error():
    assert issubclass(RegistryError, ValueError)


def test_plan_and_step_outputs_fields():
    x = torch.zeros(2, 3, 4, 4)
    plan = Plan(ctx_inputs=x, tgt_inputs=x, ctx_idx=None, tgt_idx=None)
    assert plan.ctx_idx is None and plan.tgt_idx is None
    out = StepOutputs(
        pred=torch.zeros(2, 5, 8),
        target=torch.zeros(2, 5, 8),
        emb=torch.zeros(2, 7, 8),
        progress=0.5,
    )
    assert out.progress == 0.5


def test_discover_registers_new_piece_without_touching_core(tmp_path, monkeypatch):
    pkg = tmp_path / "fake_slots"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "my_piece.py").write_text(
        textwrap.dedent(
            """
            from jepa_tools.core.registry import register

            @register("regularizer", "from_a_file")
            class FromAFile:
                pass
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        from jepa_tools.slots import discover

        discover("fake_slots")
        assert "from_a_file" in names("regularizer")
        discover("fake_slots")  # idempotent
    finally:
        for mod in [m for m in sys.modules if m.startswith("fake_slots")]:
            del sys.modules[mod]
