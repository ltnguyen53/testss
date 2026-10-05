"""Unit test cho src/training/model_setup.py bằng model giả (duck-typing, không torch).

Model giả mô phỏng backbone kiểu ResNet: input_layer, layer1..layer4 (mỗi layer có
conv + BatchNorm2d), head (fc). Chỉ kiểm tra LOGIC chọn tham số — KHÔNG chứng minh
chạy đúng với torch.nn.Module thật (xem docstring module).
"""

import pytest

from src.training.model_setup import (
    FreezePolicy,
    apply_freeze_policy,
    assert_bn_trainable,
    build_param_groups,
)


class FakeParam:
    def __init__(self, numel=1):
        self.requires_grad = True
        self._numel = numel

    def numel(self):
        return self._numel


class FakeModule:
    def __init__(self, params=None, children=None):
        self._params = dict(params or {})
        self._children = dict(children or {})
        self.training = True

    def named_children(self):
        return list(self._children.items())

    def modules(self):
        yield self
        for c in self._children.values():
            yield from c.modules()

    def parameters(self, recurse=True):
        yield from self._params.values()
        if recurse:
            for c in self._children.values():
                yield from c.parameters(recurse=True)

    def get_submodule(self, target):
        mod = self
        for part in target.split("."):
            if part not in mod._children:
                raise AttributeError(part)
            mod = mod._children[part]
        return mod


class Conv2d(FakeModule):
    def __init__(self):
        super().__init__(params={"weight": FakeParam(100)})


class BatchNorm2d(FakeModule):
    def __init__(self):
        super().__init__(params={"weight": FakeParam(10), "bias": FakeParam(10)})
        self.track_running_stats = True


class Linear(FakeModule):
    def __init__(self):
        super().__init__(params={"weight": FakeParam(50), "bias": FakeParam(5)})


def _layer():
    return FakeModule(children={"conv": Conv2d(), "bn": BatchNorm2d()})


@pytest.fixture
def model():
    return FakeModule(
        children={
            "input_layer": _layer(),
            "layer1": _layer(),
            "layer2": _layer(),
            "layer3": _layer(),
            "layer4": _layer(),
            "head": Linear(),
        }
    )


POLICY = FreezePolicy(unfreeze_modules=("layer4",), head_module="head")


def _rg(model, path):
    return [p.requires_grad for p in model.get_submodule(path).parameters()]


def test_early_conv_frozen_but_bn_affine_trainable_everywhere(model):
    apply_freeze_policy(model, POLICY)

    for name in ("input_layer", "layer1", "layer2", "layer3"):
        assert _rg(model, f"{name}.conv") == [False]  # conv weight bị freeze
        assert _rg(model, f"{name}.bn") == [True, True]  # BN gamma/beta vẫn train


def test_last_block_and_head_fully_trainable(model):
    apply_freeze_policy(model, POLICY)

    assert all(_rg(model, "layer4"))
    assert all(_rg(model, "head"))


def test_report_counts(model):
    report = apply_freeze_policy(model, POLICY)

    # 5 layer x (conv 100 + bn 20) + head 55 = 655
    assert report.total_params == 5 * 120 + 55
    # trainable = BN của 4 layer đầu (4*20) + layer4 (120) + head (55) = 255
    assert report.trainable_params == 4 * 20 + 120 + 55
    assert report.bn_affine_params == 5 * 20


def test_bn_affine_can_be_disabled(model):
    apply_freeze_policy(model, FreezePolicy(("layer4",), "head", train_bn_affine_everywhere=False))

    assert _rg(model, "layer1.bn") == [False, False]


def test_param_groups_are_disjoint_and_complete(model):
    apply_freeze_policy(model, POLICY)
    groups = build_param_groups(
        model,
        POLICY,
        head_lr=1e-3,
        backbone_unfrozen_lr=1e-5,
        bn_affine_lr=1e-5,
        weight_decay=5e-4,
    )
    by_name = {g["name"]: g for g in groups}

    assert set(by_name) == {"head", "backbone_unfrozen", "bn_affine_frozen_blocks"}
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids))  # không param nào ở 2 group
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert set(ids) == trainable_ids  # group phủ đúng mọi param trainable, không thừa/thiếu

    assert by_name["head"]["lr"] == 1e-3
    assert by_name["backbone_unfrozen"]["lr"] == 1e-5
    assert by_name["head"]["lr"] > by_name["backbone_unfrozen"]["lr"]  # discriminative LR
    assert len(by_name["bn_affine_frozen_blocks"]["params"]) == 4 * 2  # BN của 4 layer đầu
    assert by_name["bn_affine_frozen_blocks"]["weight_decay"] == 0.0
    assert by_name["head"]["weight_decay"] == 5e-4


def test_unknown_module_name_raises_with_available_names(model):
    with pytest.raises(ValueError) as exc:
        apply_freeze_policy(model, FreezePolicy(unfreeze_modules=("layer9",), head_module="head"))

    assert "layer4" in str(exc.value)  # thông báo lỗi liệt kê module có thật để sửa typo


def test_originally_frozen_param_is_never_re_enabled(model):
    """Regression test cho bug THẬT đã phát hiện + sửa (2026-09-29, xác nhận bằng
    iresnet18 thật của arcface_torch — xem docstring module điểm 4): 1 param mà
    kiến trúc GỐC đã tự đặt requires_grad=False TRƯỚC KHI apply_freeze_policy
    chạy (mô phỏng đúng `features.weight` của iresnet thật: BN cuối, weight bị
    cố định = 1.0, bias thì không) phải giữ nguyên requires_grad=False sau khi
    áp policy — dù module chứa nó nằm trong đường unfreeze BN affine. Param anh
    em (bias) trong CÙNG module đó vẫn phải được unfreeze bình thường.
    """
    features_bn = model.get_submodule("layer4.bn")  # dùng lại BatchNorm2d có sẵn trong fixture
    weight_param, bias_param = features_bn._params["weight"], features_bn._params["bias"]
    weight_param.requires_grad = False  # giả lập kiến trúc gốc tự cố định TRƯỚC khi gọi policy

    apply_freeze_policy(model, POLICY)  # layer4 nằm trong unfreeze_modules

    assert weight_param.requires_grad is False, "param bị kiến trúc gốc cố định phải giữ nguyên"
    assert bias_param.requires_grad is True, "param anh em không bị cố định vẫn phải unfreeze"

    groups = build_param_groups(
        model, POLICY, head_lr=1e-3, backbone_unfrozen_lr=1e-5, bn_affine_lr=1e-5, weight_decay=0.0
    )
    grouped_ids = {id(p) for g in groups for p in g["params"]}
    assert id(weight_param) not in grouped_ids, "param frozen không được lọt vào optimizer group"
    assert id(bias_param) in grouped_ids


def test_module_without_params_raises(model):
    model._children["act"] = FakeModule()  # module không có tham số (vd activation)

    with pytest.raises(ValueError):
        apply_freeze_policy(model, FreezePolicy(unfreeze_modules=("act",), head_module="head"))


def test_assert_bn_trainable_detects_eval_mode_and_disabled_stats(model):
    assert_bn_trainable(model)  # tất cả train mode -> không raise

    model.get_submodule("layer2.bn").training = False
    with pytest.raises(RuntimeError):
        assert_bn_trainable(model)

    model.get_submodule("layer2.bn").training = True
    model.get_submodule("layer1.bn").track_running_stats = False
    with pytest.raises(RuntimeError):
        assert_bn_trainable(model)


def test_bn_affine_exclude_modules_keeps_those_bn_frozen(model):
    policy = FreezePolicy(("layer4",), "head", bn_affine_exclude_modules=("layer1",))
    apply_freeze_policy(model, policy)

    assert _rg(model, "layer1.bn") == [False, False]  # bị loại trừ -> vẫn frozen
    assert _rg(model, "layer2.bn") == [True, True]  # các BN khác vẫn train

    groups = build_param_groups(
        model, policy, head_lr=1e-3, backbone_unfrozen_lr=1e-5, bn_affine_lr=1e-5, weight_decay=0.0
    )
    grouped_ids = {id(p) for g in groups for p in g["params"]}
    assert all(id(p) not in grouped_ids for p in model.get_submodule("layer1.bn").parameters())


def test_bn_affine_exclude_unknown_module_raises(model):
    with pytest.raises(ValueError):
        apply_freeze_policy(
            model, FreezePolicy(("layer4",), "head", bn_affine_exclude_modules=("nope",))
        )
