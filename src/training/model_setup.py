"""Freeze/unfreeze + BatchNorm + discriminative LR — SPEC v2 mục 3.1.

Chính sách:
  1. Freeze TOÀN BỘ tham số, rồi unfreeze: (a) BN affine (gamma/beta) ở MỌI block —
     kể cả block bị freeze conv, để thích nghi domain shift do occlusion mà không
     train lại conv; (b) các module trong `unfreeze_modules` (block cuối); (c) head.
     NGOẠI LỆ: param nào kiến trúc GỐC đã tự đặt requires_grad=False từ trước (vd
     `features.weight` của iresnet — xem điểm 4) thì KHÔNG bị (a)/(b)/(c) bật lại.
  2. BN running stats: cập nhật tự động khi module ở train mode — KHÔNG liên quan
     requires_grad. Bẫy kinh điển: gọi .eval() lên block "đã freeze" -> running stats
     đứng yên, mất hết tác dụng của bước (a). `assert_bn_trainable` chặn lỗi này.
  3. Discriminative LR: 3 param group tách biệt, KHÔNG trùng tham số (torch raise
     nếu 1 param nằm ở 2 group). build_param_groups lọc lại theo p.requires_grad
     hiện tại — nguồn chân lý duy nhất, không tự suy luận lại policy.
  4. BUG THẬT ĐÃ PHÁT HIỆN + SỬA (2026-09-29, xác nhận bằng iresnet18 THẬT của
     arcface_torch, không phải suy đoán): `features.weight` (scale BatchNorm1d
     cuối) bị upstream cố định = 1.0, requires_grad=False NGAY TRONG __init__
     kiến trúc gốc (lựa chọn thiết kế cố ý — embedding bị L2-normalize ngay sau
     trong ArcFace head nên scale này dư thừa). `features.bias` thì KHÔNG bị cố
     định. Trước khi sửa, policy "unfreeze mọi BN affine" sẽ VÔ TÌNH bật lại
     `features.weight` (đảo ngược đúng lựa chọn thiết kế của upstream), vì nó
     thao tác ở cấp MODULE (weight+bias cùng lúc), không phân biệt param nào đã
     bị kiến trúc gốc cố định từ trước. Xem comment chi tiết trong
     apply_freeze_policy.

Đã verify với torch.nn.Module thật (2026-09-29, torch 2.14 + iresnet18 thật tải từ
arcface_torch — không chỉ model giả trong tests/unit/test_model_setup.py): tên
module `conv1, bn1, prelu, layer1..layer4, bn2, dropout, fc, features` đúng như
giả định trong config; API dùng (modules(), named_children(), parameters(recurse=),
get_submodule(), Parameter.requires_grad/numel()) chạy đúng trên nn.Module thật,
không chỉ trên model giả duck-typed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def is_batchnorm(module: Any) -> bool:
    return "BatchNorm" in type(module).__name__


@dataclass(frozen=True)
class FreezePolicy:
    unfreeze_modules: tuple[str, ...]  # module (dot-path) unfreeze toàn bộ, vd ("layer4",)
    head_module: str  # vd "head"
    train_bn_affine_everywhere: bool = True
    # Module (dot-path) mà BN affine BỊ GIỮ frozen dù train_bn_affine_everywhere=True —
    # vd "backbone.features" nếu code gốc cố định features.weight=1 (CHƯA xác nhận, xem
    # SPEC v2.1 mục 3.1). Mặc định rỗng = không loại trừ gì.
    bn_affine_exclude_modules: tuple[str, ...] = ()


@dataclass(frozen=True)
class FreezeReport:
    total_params: int
    trainable_params: int
    bn_affine_params: int


def _get_submodule(model: Any, name: str) -> Any:
    try:
        return model.get_submodule(name)
    except AttributeError as e:
        available = [n for n, _ in model.named_children()]
        raise ValueError(
            f"Module '{name}' không tồn tại trong model. Module cấp 1 hiện có: {available}"
        ) from e


def _numel(params: list[Any]) -> int:
    return sum(p.numel() for p in params)


def _bn_affine_params(model: Any, exclude_modules: tuple[str, ...] = ()) -> list[Any]:
    excluded_ids = {
        id(m) for name in exclude_modules for m in _get_submodule(model, name).modules()
    }
    out: list[Any] = []
    for m in model.modules():
        if is_batchnorm(m) and id(m) not in excluded_ids:
            out.extend(m.parameters(recurse=False))
    return out


def apply_freeze_policy(model: Any, policy: FreezePolicy) -> FreezeReport:
    # BUG THẬT ĐÃ PHÁT HIỆN (2026-09-29, xác nhận bằng iresnet18 THẬT của
    # arcface_torch, không phải suy đoán): upstream tự đặt `features.weight`
    # (scale của BatchNorm1d cuối, embedding-level) = hằng số 1.0 VÀ
    # requires_grad=False NGAY TRONG __init__ của kiến trúc gốc — lựa chọn thiết
    # kế cố ý của ArcFace (bỏ 1 bậc tự do dư thừa vì embedding bị L2-normalize
    # ngay sau đó trong head, xem `nn.init.constant_(self.features.weight, 1.0)`
    # + dòng requires_grad=False ngay sau, trong iresnet.py gốc). CÙNG module đó,
    # `features.bias` KHÔNG bị cố định (upstream để trainable).
    #
    # Policy "unfreeze mọi BN affine" của TA hoạt động ở cấp MODULE
    # (m.parameters(recurse=False) trả về cả weight lẫn bias cùng lúc) — nếu
    # không xử lý gì thêm, ta sẽ VÔ TÌNH bật lại `features.weight`, đảo ngược
    # đúng lựa chọn thiết kế mà upstream cố ý đặt. Loại trừ theo
    # bn_affine_exclude_modules (cấp module) lại quá thô: loại cả `features.bias`
    # theo, trong khi bias NÊN được train.
    #
    # Quy tắc tổng quát để xử lý đúng, áp dụng cho MỌI bước unfreeze bên dưới
    # (không riêng BN): policy chỉ được THÊM trainability, KHÔNG BAO GIỜ ép bật
    # lại 1 param mà chính kiến trúc gốc đã cố ý đặt requires_grad=False từ TRƯỚC
    # khi ta chạm vào model. Snapshot trạng thái gốc NGAY TRƯỚC bước "freeze toàn
    # bộ" bên dưới, rồi loại các param này khỏi mọi bước unfreeze phía sau.
    originally_frozen_ids = {id(p) for p in model.parameters() if not p.requires_grad}

    for p in model.parameters():
        p.requires_grad = False

    bn_params = (
        _bn_affine_params(model, policy.bn_affine_exclude_modules)
        if policy.train_bn_affine_everywhere
        else []
    )
    bn_params = [p for p in bn_params if id(p) not in originally_frozen_ids]
    for p in bn_params:
        p.requires_grad = True

    for name in (*policy.unfreeze_modules, policy.head_module):
        candidates = list(_get_submodule(model, name).parameters())
        if not candidates:
            raise ValueError(f"Module '{name}' không có tham số nào để unfreeze")
        params = [p for p in candidates if id(p) not in originally_frozen_ids]
        if not params:
            raise ValueError(
                f"Module '{name}': mọi tham số đều bị kiến trúc gốc cố định "
                "(requires_grad=False từ trước khi apply_freeze_policy chạy) — "
                "không có gì để unfreeze thật sự. Nếu đây là chủ đích, bỏ module "
                "này khỏi unfreeze_modules/head_module."
            )
        for p in params:
            p.requires_grad = True

    all_params = list(model.parameters())
    report = FreezeReport(
        total_params=_numel(all_params),
        trainable_params=_numel([p for p in all_params if p.requires_grad]),
        bn_affine_params=_numel(bn_params),
    )
    if report.trainable_params == 0:
        raise RuntimeError("Sau khi áp freeze policy không còn tham số nào trainable")
    return report


def build_param_groups(
    model: Any,
    policy: FreezePolicy,
    *,
    head_lr: float,
    backbone_unfrozen_lr: float,
    bn_affine_lr: float,
    weight_decay: float,
    bn_affine_weight_decay: float = 0.0,
) -> list[dict[str, Any]]:
    """3 group, ưu tiên head > backbone_unfrozen > bn_affine (mỗi param đúng 1 group).

    Gọi SAU apply_freeze_policy. Trả list dict cho torch.optim (thêm key "name" —
    optimizer chấp nhận key phụ, tiện log per-group LR lên MLflow).
    """
    head_params = list(_get_submodule(model, policy.head_module).parameters())
    used = {id(p) for p in head_params}

    unfrozen: list[Any] = []
    for name in policy.unfreeze_modules:
        for p in _get_submodule(model, name).parameters():
            if id(p) not in used:
                used.add(id(p))
                unfrozen.append(p)

    bn_only = []
    if policy.train_bn_affine_everywhere:
        for p in _bn_affine_params(model, policy.bn_affine_exclude_modules):
            if id(p) not in used:
                used.add(id(p))
                bn_only.append(p)

    # BỔ SUNG (2026-09-29, cùng lúc sửa bug "features.weight" ở apply_freeze_policy):
    # lọc theo p.requires_grad TẠI ĐÂY thay vì tin tưởng mù các danh sách vừa gom —
    # nguồn chân lý duy nhất về "param nào thật sự trainable" là requires_grad hiện
    # tại của chính param đó (do apply_freeze_policy quyết định, kể cả phần loại trừ
    # "originally_frozen" mà hàm này không biết gì về nó). Nếu quên gọi
    # apply_freeze_policy trước, mọi group sẽ rỗng -> optimizer raise lỗi rõ ràng
    # ("empty parameter list") thay vì âm thầm train nhầm param bị kiến trúc gốc cố
    # định (vd features.weight).
    head_params = [p for p in head_params if p.requires_grad]
    unfrozen = [p for p in unfrozen if p.requires_grad]
    bn_only = [p for p in bn_only if p.requires_grad]

    groups = [
        {"name": "head", "params": head_params, "lr": head_lr, "weight_decay": weight_decay},
        {
            "name": "backbone_unfrozen",
            "params": unfrozen,
            "lr": backbone_unfrozen_lr,
            "weight_decay": weight_decay,
        },
        {
            "name": "bn_affine_frozen_blocks",
            "params": bn_only,
            "lr": bn_affine_lr,
            "weight_decay": bn_affine_weight_decay,
        },
    ]
    return [g for g in groups if g["params"]]


def assert_bn_trainable(model: Any) -> None:
    """Gọi sau model.train(): mọi BN phải ở train mode và track_running_stats=True.

    Fail loudly thay vì im lặng để BN đứng yên (xem docstring module, mục 2).
    """
    bad = [
        type(m).__name__
        for m in model.modules()
        if is_batchnorm(m)
        and (not getattr(m, "training", True) or not getattr(m, "track_running_stats", True))
    ]
    if bad:
        raise RuntimeError(
            f"{len(bad)} BatchNorm layer không ở train mode/track_running_stats — running stats "
            "sẽ KHÔNG cập nhật, mất tác dụng thích nghi domain shift do occlusion"
        )
