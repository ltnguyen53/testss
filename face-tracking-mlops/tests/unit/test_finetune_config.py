"""Unit test cho src/training/config.py — gồm test file YAML thật được ship trong repo
(chống lệch giữa configs/finetune_occlusion.yaml và code đọc nó)."""

import copy
from pathlib import Path

import pytest

from src.training.config import (
    freeze_policy_from_config,
    load_finetune_config,
    param_group_kwargs_from_config,
    sampler_kwargs_from_config,
    validate_finetune_config,
)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "finetune_occlusion.yaml"


@pytest.fixture
def cfg():
    return load_finetune_config(CONFIG_PATH)


def test_shipped_config_loads_and_validates(cfg):
    assert cfg["freeze"]["train_bn_affine_everywhere"] is True
    assert cfg["optimizer"]["head_lr"] > cfg["optimizer"]["backbone_unfrozen_lr"]


def test_helpers_build_expected_objects(cfg):
    policy = freeze_policy_from_config(cfg)
    assert policy.unfreeze_modules == ("backbone.layer4",)  # fc mặc định KHÔNG được mở
    assert policy.head_module == "head"
    assert policy.bn_affine_exclude_modules == ()

    assert set(param_group_kwargs_from_config(cfg)) == {
        "head_lr",
        "backbone_unfrozen_lr",
        "bn_affine_lr",
        "weight_decay",
        "bn_affine_weight_decay",
    }
    sk = sampler_kwargs_from_config(cfg)
    assert sk["seed"] == cfg["seed"]
    assert sk["images_per_identity"] == 4


def test_missing_section_and_missing_key_raise(cfg):
    bad = copy.deepcopy(cfg)
    del bad["sampler"]
    with pytest.raises(ValueError):
        validate_finetune_config(bad)

    bad = copy.deepcopy(cfg)
    del bad["optimizer"]["head_lr"]
    with pytest.raises(ValueError):
        validate_finetune_config(bad)


def test_non_discriminative_lr_raises(cfg):
    bad = copy.deepcopy(cfg)
    bad["optimizer"]["head_lr"] = bad["optimizer"]["backbone_unfrozen_lr"]  # dùng chung 1 LR
    with pytest.raises(ValueError):
        validate_finetune_config(bad)


def test_invalid_margin_and_sampler_raise(cfg):
    bad = copy.deepcopy(cfg)
    bad["head"]["arcface_margin"] = 1.5
    with pytest.raises(ValueError):
        validate_finetune_config(bad)

    bad = copy.deepcopy(cfg)
    bad["sampler"]["masked_per_identity"] = bad["sampler"]["images_per_identity"] + 1
    with pytest.raises(ValueError):
        validate_finetune_config(bad)


def test_train_embedding_fc_flag_adds_fc_and_exclude_gets_prefixed(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["freeze"]["train_embedding_fc"] = True
    cfg["freeze"]["bn_affine_exclude_modules"] = ["features"]

    policy = freeze_policy_from_config(cfg)

    assert policy.unfreeze_modules == ("backbone.layer4", "backbone.fc")
    assert policy.bn_affine_exclude_modules == ("backbone.features",)


def test_fc_in_trainable_blocks_raises(cfg):
    bad = copy.deepcopy(cfg)
    bad["freeze"]["trainable_blocks"] = ["layer4", "fc"]  # né cờ train_embedding_fc
    with pytest.raises(ValueError):
        validate_finetune_config(bad)

    bad = copy.deepcopy(cfg)
    bad["freeze"]["trainable_blocks"] = []
    with pytest.raises(ValueError):
        validate_finetune_config(bad)
