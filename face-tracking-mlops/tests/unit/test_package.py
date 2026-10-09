"""Phase 0 sanity test.

Chưa có logic thật để test — mục đích duy nhất của test này là cho CI
có 1 test THẬT để chạy (tránh pytest exit code 5 "no tests collected",
đồng thời verify package `src/` import được đúng theo cấu trúc mục 1 spec).
Xoá/thay bằng test thật khi Phase 1+ có code thật trong các subpackage.
"""

import importlib

SUBPACKAGES = [
    "src",
    "src.data",
    "src.training",
    "src.export",
    "src.api",
    "src.evaluation",
]


def test_src_subpackages_import():
    for name in SUBPACKAGES:
        importlib.import_module(name)
