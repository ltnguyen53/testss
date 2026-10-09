"""Checkpoint save/resume — mục 2.2 spec: atomic write, giữ tối thiểu N bản gần
nhất, resume chỉ khi config_hash khớp.

Tách biệt CỐ Ý khỏi torch: serialization thật (torch.save/torch.load) được inject
qua save_fn/load_fn — cho phép test toàn bộ logic file-management (atomic write,
rotation, resume decision, config hash matching) bằng fake serializer, không cần
torch cài trong môi trường viết logic thuần. Mặc định dùng torch.save/torch.load
thật khi không truyền save_fn/load_fn — ĐÃ VERIFY chạy thật với torch 2.14
(2026-09-29, xem test_checkpoint_default_save_load_roundtrip_with_real_torch
trong tests/unit/test_checkpoint.py): phát hiện và sửa 1 bug thật trong lúc
verify — xem comment trong _default_torch_load về weights_only.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


def compute_config_hash(config: dict) -> str:
    """Hash ổn định của config, không phụ thuộc thứ tự key trong file yaml.

    Dùng để xác định 1 checkpoint có "khớp" config hiện tại hay không — sinh ra
    từ config khác thì KHÔNG được resume nhầm vào (silent bug nếu resume nhầm:
    optimizer/scheduler state không tương thích kiến trúc/hyperparameter mới).

    BỔ SUNG (2026-09-29, sửa lúc train.py lần đầu THẬT SỰ gọi hàm này với full
    config — trước đó chỉ có test gọi với dict nhỏ tự tạo, chưa lộ vấn đề):
    LOẠI các section/field thuần vận hành (KHÔNG ảnh hưởng gradient/optimization
    semantics) khỏi hash — xem CONFIG_HASH_EXCLUDE_PATHS ngay dưới. Lý do: đổi
    `checkpoint.dir`/`mlflow.experiment_name` không đổi GÌ về việc đang train
    cái gì, chỉ đổi CÁCH lưu/log. Nếu tính vào hash, đổi tần suất save hay tên
    MLflow experiment sẽ vô tình đổi config_hash, khiến find_resumable() không
    tìm thấy checkpoint cũ dù model/data/hyperparameter không đổi gì.

    `training.epochs` CŨNG bị loại (phát hiện bằng test tích hợp thật —
    tests/unit/test_train.py:test_train_resumes_from_a_fresh_process_instead_of_restarting
    — ban đầu để cả section "training" vào hash, khiến việc resume để TRAIN
    THÊM epoch sau khi rớt mạng, giống hệt kịch bản Colab thật, bị chặn nhầm):
    epochs chỉ là tiêu chí DỪNG vòng lặp, không ảnh hưởng gradient nào được
    tính ở bất kỳ step nào (project hiện không có LR schedule phụ thuộc epoch)
    — khác optimizer_type/momentum (vẫn nằm trong hash vì ảnh hưởng thật tới
    việc tính gradient/update).
    """
    import hashlib

    exclude_paths = ("checkpoint", "mlflow", "training.epochs")
    relevant = _drop_paths(config, exclude_paths)
    canonical = json.dumps(relevant, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _drop_paths(config: dict, dotted_paths: tuple[str, ...]) -> dict:
    """Trả về BẢN SAO config, loại các path dạng "a" (cả section) hoặc "a.b"
    (1 field trong section) — không sửa config gốc (caller có thể vẫn cần
    dùng nguyên bản sau khi gọi hàm này)."""
    top_level_drop = {p for p in dotted_paths if "." not in p}
    nested_drop: dict[str, set[str]] = {}
    for p in dotted_paths:
        if "." in p:
            section, field = p.split(".", 1)
            nested_drop.setdefault(section, set()).add(field)

    out: dict = {}
    for key, value in config.items():
        if key in top_level_drop:
            continue
        if key in nested_drop and isinstance(value, dict):
            out[key] = {k: v for k, v in value.items() if k not in nested_drop[key]}
        else:
            out[key] = value
    return out


def get_git_commit_hash(repo_root: Path = Path(".")) -> str:
    """Short commit hash hiện tại, hoặc "unknown" nếu không lấy được (vd chưa
    git init, hoặc chạy trong container không có .git) — KHÔNG raise, vì thiếu
    thông tin này không nên chặn training chạy.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001 — cố ý bắt rộng, đây là best-effort metadata
        return "unknown"


@dataclass(frozen=True)
class CheckpointMeta:
    timestamp: str  # ISO8601, UTC
    config_hash: str
    git_commit_hash: str
    epoch: int
    step: int


def _default_torch_save(state: Any, path: Path) -> None:
    import torch  # import trễ — module này import được cả khi không có torch

    torch.save(state, path)


def _default_torch_load(path: Path) -> Any:
    import torch

    # BUG ĐÃ SỬA (2026-09-29, đã verify thật với torch 2.14 — không còn là rủi ro
    # "chưa verify"): từ torch 2.6, mặc định weights_only=True, unpickler an toàn
    # của nó KHÔNG cho phép nhiều kiểu dữ liệu checkpoint tự sinh của ta chứa (vd
    # numpy.random.get_state() trả về tuple lồng numpy.ndarray — xem rng_state.py).
    # Load mặc định (không truyền weights_only) FAIL ngay ở numpy tuple này (đã
    # test thật: torch.load ném UnpicklingError). weights_only=False AN TOÀN ở
    # đây vì checkpoint này LUÔN do CHÍNH pipeline này tự ghi ra (Google Drive
    # của chính người dùng), không phải file tải về từ nguồn không tin cậy —
    # khác hẳn trường hợp weights_only=True được thiết kế để phòng (load
    # checkpoint pretrained tải từ Internet).
    return torch.load(path, map_location="cpu", weights_only=False)


class CheckpointManager:
    """Quản lý checkpoint trên Google Drive (hoặc bất kỳ thư mục nào — không biết
    gì về Drive cụ thể, chỉ nhận 1 Path, đúng nguyên tắc code thuần không biết hạ
    tầng — việc mount Drive là việc của notebook, không phải của class này).
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        keep_last_n: int = 2,
        save_fn: Callable[[Any, Path], None] | None = None,
        load_fn: Callable[[Path], Any] | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last_n = keep_last_n
        self._save_fn = save_fn or _default_torch_save
        self._load_fn = load_fn or _default_torch_load

    def save(self, state: Any, meta: CheckpointMeta) -> Path:
        """Ghi checkpoint mới: state (qua save_fn) + meta.json, qua thư mục .tmp
        rồi rename atomic — không bao giờ để lại checkpoint dở dang nếu bị ngắt
        giữa chừng (rớt mạng Colab, hết quota GPU).

        Trả về Path thư mục checkpoint vừa ghi xong.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_name = f"ckpt_epoch{meta.epoch:04d}_step{meta.step:08d}"
        final_dir = self.checkpoint_dir / ckpt_name
        tmp_dir = self.checkpoint_dir / f".tmp_{ckpt_name}"

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        self._save_fn(state, tmp_dir / "state.ckpt")
        (tmp_dir / "meta.json").write_text(
            json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if final_dir.exists():
            shutil.rmtree(final_dir)
        tmp_dir.rename(final_dir)  # atomic ở cấp filesystem (không phải copy-rồi-xoá)

        # BUG ĐÃ SỬA (2026-09-28): rotate KHÔNG được xoay vòng chung giữa nhiều
        # config_hash trong cùng checkpoint_dir. Trước đây _rotate() sort TOÀN BỘ
        # checkpoint (mọi hash) theo tên thư mục rồi giữ N bản "mới nhất theo tên".
        # Vì tên thư mục bắt đầu lại từ epoch0000 mỗi khi đổi config (hash khác),
        # 1 run mới (config_hash khác, epoch thấp) có thể bị coi là "cũ hơn" 1 run
        # cũ (config_hash khác, epoch cao) đang còn nằm trên Drive từ trước — dẫn
        # tới checkpoint VỪA LƯU của run mới bị xoá ngay lập tức, find_resumable()
        # sau đó luôn trả None. Đã tái hiện lỗi này bằng script thực nghiệm trước
        # khi sửa. Fix: mỗi config_hash có cửa sổ rotate RIÊNG (keep_last_n bản
        # gần nhất CỦA CHÍNH hash đó) — checkpoint của hash khác không bị đụng tới.
        self._rotate(meta.config_hash)
        return final_dir

    def _list_checkpoints(
        self, config_hash: str | None = None
    ) -> list[tuple[Path, CheckpointMeta]]:
        """Trả về [(dir, meta)] đã đọc meta.json, sort theo (epoch, step) — KHÔNG
        sort theo tên thư mục (tên chỉ trùng epoch/step khi cùng 1 config_hash;
        xem giải thích trong save()). Lọc theo config_hash nếu được truyền."""
        if not self.checkpoint_dir.exists():
            return []
        out: list[tuple[Path, CheckpointMeta]] = []
        for p in self.checkpoint_dir.iterdir():
            if not (p.is_dir() and p.name.startswith("ckpt_") and (p / "meta.json").exists()):
                continue
            meta = self._read_meta(p)
            if meta is None:
                continue
            if config_hash is not None and meta.config_hash != config_hash:
                continue
            out.append((p, meta))
        out.sort(key=lambda item: (item[1].epoch, item[1].step))
        return out

    def _rotate(self, config_hash: str) -> None:
        """Chỉ xoay vòng trong phạm vi CÙNG config_hash — checkpoint của hash
        khác (run khác) trên cùng checkpoint_dir không bị xoá bởi rotate này."""
        ckpts = self._list_checkpoints(config_hash=config_hash)
        while len(ckpts) > self.keep_last_n:
            oldest_dir, _ = ckpts.pop(0)
            shutil.rmtree(oldest_dir, ignore_errors=True)

    def _read_meta(self, ckpt_dir: Path) -> CheckpointMeta | None:
        try:
            raw = json.loads((ckpt_dir / "meta.json").read_text(encoding="utf-8"))
            return CheckpointMeta(**raw)
        except Exception:  # noqa: BLE001
            logger.warning("meta.json hỏng/không hợp lệ, bỏ qua checkpoint: %s", ckpt_dir)
            return None

    def find_resumable(self, config_hash: str) -> Path | None:
        """Checkpoint MỚI NHẤT (epoch, step lớn nhất) có config_hash khớp config
        hiện tại. Không khớp -> None (train từ đầu/từ pretrained) — KHÔNG bao giờ
        resume vào checkpoint sinh ra từ config khác, dù đó là checkpoint có
        epoch/step lớn nhất trên toàn bộ Drive.
        """
        matching = self._list_checkpoints(config_hash=config_hash)
        if not matching:
            return None
        return matching[-1][0]  # đã sort theo (epoch, step) tăng dần trong _list_checkpoints

    def load(self, ckpt_dir: Path) -> tuple[Any, CheckpointMeta]:
        meta = self._read_meta(ckpt_dir)
        if meta is None:
            raise ValueError(f"Checkpoint hỏng, không có meta.json hợp lệ: {ckpt_dir}")
        state = self._load_fn(ckpt_dir / "state.ckpt")
        return state, meta
