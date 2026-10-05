# face-tracking-mlops

Occlusion-aware face detection + tracking. Đặc tả: `face-tracking-mlops-spec` **v2.1** (giữ ở nơi bạn lưu spec, không copy vào repo). v2.1 = v2 + chốt backbone iresnet `arcface_torch`, mục Pretrained weights, tên module thật, rủi ro kích thước model, hợp đồng preprocessing.

> **Phạm vi**: đây là project **demo/portfolio học tập**, không phải hệ thống biometric triển khai thật, không dùng cho bảo mật/giám sát. Server chỉ lưu embedding (không lưu ảnh gốc), có cơ chế xoá đăng ký. Chi tiết: `docs/privacy.md` (Phase 6, chưa viết).

## Chạy local

```bash
pip install -e .                 # SPEC 11.2 — để `import src.*` chạy được từ mọi thư mục
pip install pytest pyyaml ruff numpy
pytest                           # mặc định chỉ tests/unit (xem pyproject)
```

## Trạng thái theo Phase (đánh số theo SPEC v2 mục 16)

| Phase | Trạng thái |
|---|---|
| 0 Bootstrap | Code/config xong. **Chưa đạt DoD** — cần bạn chạy các bước "Việc bạn phải tự làm" bên dưới (lock file, DVC remote, `pre-commit run --all-files`). |
| 1 Data | Code + test xong (`labels.py`, `preprocessing.py`, `dvc.yaml`, `tests/unit/test_data_split.py`). **Chưa chạy trên RMFD thật.** |
| 2 Training + Registry | **Code xong hết phần local-train**: `head.py` (ArcFace), `model.py` (`FaceModel`), `dataset.py` (load_batch), `train.py` (entrypoint đầy đủ — freeze/param-groups/resume/class-mean init/checkpoint/training loop), `mlflow_logging.py` (thin wrapper), `notebooks/train_colab.ipynb` (đã tạo, 2026-10-04 — run-sheet đúng SPEC mục 10: mount Drive, cài đặt, `dvc pull`, 1 lệnh `train.py`, 1 lệnh `run_eval.py`, `dvc push`/`git push` — đã validate bằng `nbformat.validate`, chưa chạy thật trên Colab thật). Checkpoint/resume, RNG state, PK sampler + fallback synthetic, freeze policy + discriminative LR + BN, config có validate, quản lý weight (SOURCE.yaml + SHA256), loader backbone iresnet, hợp đồng preprocessing — **đã có từ trước, giữ nguyên**. Model registry (`mlflow.register_model`) chuyển sang Phase 3 (xem lý do trong `src/evaluation/registry.py` — cần số đo verification thật mới quyết định đăng ký được, `train.py` chỉ có training loss). **Chưa chạy thật trên RMFD + backbone r50 thật** (mọi test Phase 2 dùng backbone giả nhỏ + data giả). |
| 3 Evaluation + Threshold | **Code xong**: `embed.py` (bỏ head, chỉ embedding — torch), `pairs.py` (genuine/impostor 3 nhóm occlusion, loại synthetic khỏi masked_masked — đóng nợ SPEC "B6", thuần Python), `threshold.py` (TAR@FAR sweep + khoá, thuần Python), `report.py` (sinh `reports/eval_metrics.json`, NaN -> null cho JSON chuẩn), `registry.py` (logic đăng ký, thuần Python — KHÔNG tự promote Production, luôn cần review thủ công theo SPEC 5), `run_eval.py` (entrypoint nối toàn bộ + `configs/inference.yaml`). Threshold TOÀN CỤC (sweep trên pool cả 3 nhóm của val), re-validate trên test bằng đúng threshold đã khoá, không tinh chỉnh lại. **Chưa có**: benchmark MLFW/OCFR-2022 (chỉ mới dùng val/test tự có), chạy thật trên RMFD (test Phase 3 dùng data giả + backbone giả, giống Phase 2). |
| 4–7 | Chưa bắt đầu — **ngoại lệ**: đã thêm job CI `train-test` (chạy lại `pytest tests/unit` với torch cài đủ, đóng khoảng trống "CI không chứng minh được gì về phần ML" — xem mục 5 ở "Chỗ CỐ Ý lệch so với spec" và phần "Test" ngay dưới). Đây KHÔNG phải tiến độ Phase 5 thật (chưa có `Dockerfile`/`src/api`), chỉ là sửa 1 lỗ hổng hạ tầng phát hiện được trong lúc review. |

**Test: 141 unit test PASS THẬT** (torch 2.14, pytest, ruff — không phải runner tự viết) trong phiên review/sửa lỗi sau khi code được giao, gồm 2 test tích hợp chạy thật (`train.py`: vài step + resume qua 2 lần gọi độc lập; `run_eval.py`: embed → pairs → sweep threshold → report → đăng ký model, qua checkpoint thật do `CheckpointManager` ghi). **8 bug thật được phát hiện + sửa trong phiên review này** (không phải suy đoán — mỗi bug có test tái hiện trước khi sửa): (1) `_rotate` xoá nhầm checkpoint khác config_hash; (2) MaskTheFace subprocess thiếu `cwd`; (3) `torch.load` mặc định fail với checkpoint RNG state (torch ≥2.6 đổi default `weights_only`); (4) `apply_freeze_policy` vô tình bật lại `features.weight` dù kiến trúc gốc iresnet cố định nó — xác nhận bằng backbone `r18` thật tải từ `arcface_torch`; (5) `compute_config_hash` tính cả `training.epochs` vào hash, chặn resume khi chỉ muốn train thêm epoch; (6) `_run_training_loop`/mlflow run mở quá trễ trong `train()`, lỗi setup không được ghi nhận; (7) `pairs.py` đếm genuine pair 2 lần (`(a,b)` và `(b,a)`) khi 2 ảnh cùng nguồn (unmasked_unmasked/masked_masked); (8) `pairs.py` crash khi 1 trong 2 pool impostor rỗng (điều kiện tiền kiểm tra sai). **CẬP NHẬT 2026-10-02**: CI giờ có **3 job** — `lock-sync`, `lint-test` (không cài torch — mọi test cần torch tự skip ở đây qua `pytest.importorskip("torch")`), và **`train-test`** (mới thêm — cài đủ `requirements/train.txt`, chạy lại `pytest tests/unit`, lúc này 26 test cần torch mới thật sự chạy trong CI, không chỉ trên máy review). `pairs.py`/`threshold.py`/`report.py`/`registry.py` không cần torch nên chạy ở cả 2 job `*-test`. `train-test` hiện **chưa chạy được trên GitHub thật** — chưa push lên GitHub để kiểm chứng, và cần `requirements/train.txt` được commit trước (xem "Việc bạn phải tự làm" mục 1 — `train.txt` chưa sinh được, hết đĩa giữa chừng khi thử trong sandbox viết code này).

## Chỗ CỐ Ý lệch so với spec v2 (đọc trước khi tin file trong repo)

1. **`pyproject.toml` — `packages.find`**: spec viết `where = ["src"]`. Đã chạy thử `find_packages`: cách này chỉ expose `data`, `training`, `api`… (top-level), **không** expose `src.*`, trong khi toàn bộ code import `from src.…` và spec yêu cầu `python -m src.training.train`. Dùng `where = ["."]` + `include = ["src*"]`; đã verify `pip install -e .` rồi `import src.data.labels` từ `cwd=/tmp` chạy được. Thêm `[build-system]` (spec không có) cho tường minh.
2. **Kiểm tra lock-sync (pre-commit + CI)**: snippet spec compile ra `/tmp/api.txt` rồi `diff`. Sẽ luôn fail vì thiếu `--generate-hashes` trong khi spec 11.1 yêu cầu lock có hash. Thay bằng: 1 lệnh canonical duy nhất, compile tại chỗ, rồi `git diff --exit-code`:
   `pip-compile --generate-hashes --no-header requirements/api.in -o requirements/api.txt` (tương tự `train`).
   Pre-commit chỉ kiểm `api` và chỉ khi file `requirements/api.*` đổi (train compile tốn vài phút mỗi commit); CI kiểm cả hai.
   **ĐÃ VERIFY (2026-10-02, chạy pip-compile thật)**: `--no-header --generate-hashes` cho kết quả **idempotent** (chạy lại nhiều lần ra cùng 1 file byte-for-byte) — xác nhận cách tiếp cận "1 lệnh canonical + diff" ở trên hoạt động đúng như thiết kế, không chỉ là giả định.
3. **`requirements/train.in`**: bỏ `insightface`. Theo hiểu biết của tôi (chưa verify) pip package đó chủ yếu phục vụ inference ONNX, không phải backbone PyTorch để fine-tune.
4. `tests/unit` mặc định; `tests/integration` (ONNX) chạy tường minh ở CI job 3 (Phase 5).
5. **CI có thêm job `train-test`** (không có trong SPEC v2.1 mục 12 gốc — chỉ liệt kê 4 job cho Phase 5, không có job nào chạy lại test với torch cài đủ). Lý do: `lint-test` (job khớp SPEC) cố ý không cài torch nên bỏ qua ~1/5 số test — gồm 2 test tích hợp quan trọng nhất (`train.py`, `run_eval.py` chạy thật). `train-test` chạy lại chính xác `pytest tests/unit` với `requirements/train.txt` cài đủ, không phải job thay thế — đây là bổ sung cho SPEC, không mâu thuẫn với nó. Rủi ro disk/tốc độ của job này: xem comment trong `ci.yml`.

## Việc bạn phải tự làm (cần mạng — sandbox không làm được)

**1. Lock file** — dùng **cùng Python minor với CI** (`PYTHON_VERSION` trong `.github/workflows/ci.yml`, hiện 3.11), nếu không lock-sync đỏ dù `.in` không đổi:

```bash
pip install pip-tools
pip-compile --generate-hashes --no-header requirements/api.in -o requirements/api.txt
pip-compile --generate-hashes --no-header requirements/train.in -o requirements/train.txt
```

`requirements/api.txt` **đã sinh thật và commit sẵn trong repo** (2026-10-02, chạy `pip-compile` thật, đã verify idempotent — chạy lại cho kết quả giống hệt, `lock-sync` sẽ pass) — không cần làm lại trừ khi sửa `api.in`.

`requirements/train.txt` **CHƯA sinh được** — đã thử thật trong sandbox viết code này và **hết dung lượng đĩa giữa chừng** (`pip-compile` cho `torch`/`torchvision` cần tải wheel thật để resolve, không chỉ metadata như gói nhẹ — khác hẳn `api.txt`, nơi resolve chỉ cần metadata nên nhẹ). Chạy lệnh thứ 2 ở trên trên máy có đủ đĩa (khuyến nghị: ngay trên Colab, vừa đủ đĩa vừa đúng Python version Colab dùng để train thật) rồi commit `requirements/train.txt`. Rủi ro đã biết thêm: Python của Colab có thể khác 3.11 (bản torch pin theo Python version) — nếu lệch, sinh `train.txt` theo đúng Python của Colab rồi đổi `PYTHON_VERSION` của CI cho khớp (ảnh hưởng cả `train-test` job mới thêm, không chỉ `lock-sync`).

**2. Pre-commit** (DoD Phase 0): `pip install pre-commit && pre-commit install && pre-commit run --all-files`. File trong repo được viết tay, **chưa qua `ruff format`** — lần chạy đầu nhiều khả năng ruff tự sửa vài file (xuống dòng…) và báo fail; `git add` phần đã sửa rồi chạy lại.

**3. Git + DagsHub + DVC:**

```bash
git init && git add -A && git commit -m "bootstrap" && git remote add origin <URL> && git push -u origin main
pip install "dvc[s3]" && dvc init
dvc remote add -d dagshub-storage s3://dvc
dvc remote modify dagshub-storage endpointurl https://dagshub.com/<user>/<repo>.s3
dvc remote modify dagshub-storage --local access_key_id <dagshub_token>
dvc remote modify dagshub-storage --local secret_access_key <dagshub_token>
dvc push   # rỗng lúc này = DoD "dvc pull/push chạy được rỗng"
```

**4. Data (Phase 1)** — RMFD từ repo `X-zhangyang/Real-World-Masked-Face-Dataset` (README tiếng Trung, link Baidu/Google Drive). Sắp xếp lại thành `data/raw/rmfd/masked/<identity_id>/*.jpg` và `data/raw/rmfd/unmasked/<identity_id>/*.jpg`; cấu trúc gốc **chưa chắc** đúng convention này — tự kiểm tra. Ảnh cá nhân: `data/raw/enrollment/<ten>/*.jpg`.

**5. MaskTheFace**: `git submodule add https://github.com/aqeelanwar/MaskTheFace.git third_party/MaskTheFace && pip install -r third_party/MaskTheFace/requirements.txt`. Chạy `python -m src.data.preprocessing --config configs/data.yaml --dry-run` trước khi chạy thật.

## Secrets (SPEC 11.3)

`.env` thật **không bao giờ commit**; chỉ commit `.env.example`. Secret thật đặt ở:
- **GitHub Actions secrets**: `DAGSHUB_TOKEN`, `RENDER_DEPLOY_HOOK_URL`, token GHCR (dùng từ Phase 5).
- **Render dashboard → Environment**: `MLFLOW_TRACKING_URI`, `DAGSHUB_TOKEN`, `MODEL_REGISTRY_NAME`, `CORS_ALLOWED_ORIGINS`, `LOG_LEVEL`.
- **Colab**: Colab Secrets hoặc cell đầu notebook, cho `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`.

## Backbone: iresnet của `arcface_torch` (đã chốt) — việc bạn phải làm để dùng được

```bash
# 1. Lấy code backbone (sparse clone, không kéo cả repo insightface)
git clone --depth 1 --filter=blob:none --sparse https://github.com/deepinsight/insightface.git third_party/insightface
git -C third_party/insightface sparse-checkout set recognition/arcface_torch
git -C third_party/insightface rev-parse HEAD          # ghi vào upstream_commit trong SOURCE.yaml

# 2. Tải TAY backbone.pth từ model zoo của arcface_torch (Baidu/OneDrive, README repo có link) -> models/pretrained/backbone.pth
sha256sum models/pretrained/backbone.pth               # ghi vào sha256 trong SOURCE.yaml; điền nốt model_name, download_date
dvc add models/pretrained/backbone.pth && git add models/pretrained/backbone.pth.dvc models/pretrained/SOURCE.yaml
```

Các điểm dưới đây **ĐÃ xác nhận bằng torch thật** trong phiên review/sửa lỗi (2026-09-29, torch 2.14, tải `backbones/iresnet.py` thật từ `arcface_torch`, instantiate `r18` với random init — KHÔNG phải weight pretrained thật, chỉ để kiểm tra kiến trúc/API):

1. ✅ Tên module thật khớp đúng config: `conv1, bn1, prelu, layer1..layer4, bn2, dropout, fc, features`.
2. ✅ `features.weight` BỊ cố định `requires_grad=False` trong `__init__` gốc (scale BN cuối = hằng số 1.0, lựa chọn thiết kế cố ý — embedding bị L2-normalize ngay sau trong ArcFace head). `features.bias` KHÔNG bị cố định. **Đã sửa bug thật phát sinh từ phát hiện này**: `apply_freeze_policy` (model_setup.py) từng vô tình bật lại `features.weight` khi áp policy "unfreeze mọi BN affine" — giờ tự động giữ nguyên mọi param kiến trúc gốc đã tự frozen, không cần khai `bn_affine_exclude_modules: ["features"]` thủ công nữa.
3. ✅ `import backbones` CHỈ cần `torch` — không đòi `timm` (chỉ lazy-import khi dùng `vit`, project này không dùng).
4. ✅ `fc` đúng 12.845.568 tham số — khớp tính toán trong code (512×7×7×512+512).
5. **Vẫn CHƯA xác nhận** (cần `backbone.pth` thật, không tải được trong sandbox này — xem mục "Dữ liệu/weight thật" ngay dưới): `load_state_dict(strict=True)` với weight pretrained thật. **Đã xác nhận thêm (2026-10-03)**, dù vẫn random-init: forward + backward THẬT chạy đúng trên backbone `r18` thật (không phải giả lập) với ẢNH THẬT (không phải ảnh giả/ô màu) — `layer4` nhận gradient, `embed_manifest` cho ra embedding L2-norm=1.0 và KHÁC nhau giữa các ảnh khác nhau (không bị collapse). Đây là test tay (ad-hoc), KHÔNG nằm trong `tests/unit` — xem lý do (chất lượng dữ liệu nguồn) ở mục dưới.
6. **Phát hiện thêm (2026-10-03)**: `iresnet.py` gốc của `arcface_torch` gọi `torch.cuda.amp.autocast(self.fp16)` — API đã deprecated từ các bản torch gần đây, torch 2.14 báo `FutureWarning: torch.cuda.amp.autocast(args...) is deprecated. Please use torch.amp.autocast('cuda', args...) instead`. Không phải lỗi của code project này (nằm trong source upstream, không sửa được trừ khi fork/vá lại), nhưng đáng ghi chú vì `FutureWarning` thường là dấu hiệu API sẽ bị xoá hẳn ở bản torch tương lai — có thể cần vá (monkeypatch hoặc fork cục bộ 1 dòng) nếu 1 bản torch sau này xoá hẳn `torch.cuda.amp.autocast`.

Rủi ro còn mở (không phải quyết định của code): kích thước file r50 trên browser (đo sau export INT8, xem spec v2.1 mục 6) và quy tắc alignment khuôn mặt (`alignment: null` trong `configs/export.yaml` — `src/training/dataset.py` hiện chỉ resize thô, KHÔNG align, ghi rõ trong docstring). Weight chỉ dùng phi thương mại — ghi vào model card.

## Dữ liệu/weight thật — kết quả thử nghiêm túc trong phiên review (2026-10-03)

Đã chủ động thử lấy từng loại dữ liệu/weight thật mà project cần, trong giới hạn mạng của sandbox viết code này (chỉ GitHub/PyPI/npm/crates/Ubuntu archive — **KHÔNG có Baidu/OneDrive/Google Drive/Dropbox**). Ghi lại đầy đủ để bạn không tốn thời gian thử lại đúng những hướng đã biết là bế tắc.

**KHÔNG lấy được — chỉ phát hành qua kênh ngoài allowlist mạng:**
- **Weight pretrained ArcFace thật** (r18/r34/r50/r100, MS1MV2/MS1MV3/Glint360K): toàn bộ model zoo chính thức của `arcface_torch` chỉ phát hành qua Baidu Pan/OneDrive — đã tìm kỹ, không có bản mirror nào trên GitHub Releases.
- **RMFD gốc** (paired masked+unmasked, đúng 525 identity như README chính thức mô tả): chỉ qua Baidu Pan.
- **MLFW**: trang chính thức `whdeng.cn/mlfw` không trong allowlist.
- **OCFR-2022**: code + protocol đánh giá có trên GitHub (`NetoPedro/OCFR-2022`, đã đọc được README thật) nhưng **ảnh thật chỉ phát hành qua Google Drive** — repo GitHub chỉ có script, không có data.
- **ROF (RealWorldOccludedFaces)**: repo GitHub tồn tại, cấu trúc đúng ý (neutral/masked/sunglasses, có identity thật) nhưng README ghi rõ "dataset sẽ release sau" — tự kiểm tra `download.py` xác nhận nó đọc từ thư mục cục bộ `ROF/` chưa từng được publish, không phải lỗi thao tác.
- **dlib — ĐÃ GIẢI QUYẾT (2026-10-03, cập nhật so với lần thử trước)**: gói `dlib` chính thức trên PyPI chỉ có source distribution (`dlib-20.0.1.tar.gz`, không có wheel dựng sẵn cho Python 3.12) — compile từ source tốn disk không lường trước được nên lần thử trước chủ động bỏ qua. Lần này tìm ra **`dlib-bin`** (gói PyPI khác, bên thứ 3, build sẵn wheel cho nhiều platform/Python version — đã verify: cp312 Linux, 4.2MB, cài xong `import dlib` chạy ngay, không cần cmake/compile). `pip install dlib-bin` **thay cho** dòng `dlib==19.19.0` trong `third_party/MaskTheFace/requirements.txt` (bản pin 2020 đó không có wheel hiện đại) — cài các dep còn lại của file đó bình thường (`dotmap`, `imutils`, `opencv-python`/`opencv-python-headless`, `tqdm` — đều nhẹ, có wheel sẵn). **Đã chạy MaskTheFace THẬT thành công** bằng tổ hợp này — xem mục dưới.

**Lấy được — dùng để test thật, kèm cảnh báo chất lượng:**
- 1 mirror GitHub KHÔNG CHÍNH THỨC của RMFD (`X-zhangyang/Real-World-Masked-Face-Dataset`, nhánh `RWMFD_part_1`): 24 thư mục "identity", 1194 ảnh `.jpg` thật (31MB, tải qua `codeload.github.com`). **CẢNH BÁO CHẤT LƯỢNG — đã spot-check bằng mắt, không phải suy đoán**: thư mục `0001` lẫn ảnh quảng cáo khẩu trang (không phải ảnh người thật chụp), thư mục `0002` là ảnh đám đông (hàng chục người, không phải 1 identity). Chỉ thư mục `0000` (4 ảnh) được verify bằng mắt là nhất quán 1 người thật xuyên suốt. **Khuyến nghị: KHÔNG dùng mirror này làm nguồn RMFD chính thức cho project** — ít nhất phải tự lọc lại bằng tay hoặc bằng face-detection trước khi tin cậy, hoặc tải bản gốc qua Baidu Pan như README chính thức hướng dẫn.
- Dùng 4 ảnh thật của thư mục `0000` (đã verify) chạy qua **toàn bộ pipeline thật** — xem mục 5 ở trên.
- **MaskTheFace THẬT — đã chạy thành công (2026-10-03, không phải fake runner trong test)**: cài `dlib-bin` + `dotmap` + `imutils` + `opencv-python-headless` (đều có wheel sẵn, nhẹ — xem mục dlib ở trên), tải repo `aqeelanwar/MaskTheFace` thật qua `codeload.github.com`. Phát hiện quan trọng: **`download_dlib_model()` tự động của chính MaskTheFace FAIL** trong sandbox này — hàm đó tải từ `http://dlib.net/...`, domain này **không nằm trong allowlist mạng** (khác GitHub/PyPI), nên tải về 0 byte và lỗi `bz2.OSError: Invalid data stream` ngay bước giải nén. Comment cũ trong `configs/data.yaml` ("chỉ cần Colab có mạng") có thể đúng cho Colab (có internet không giới hạn) nhưng SAI nếu mạng bị hạn chế domain như sandbox này — đã sửa comment đó để hedge đúng mức, kèm hướng dẫn fallback tải tay qua GitHub mirror (`davisking/dlib-models`, đã verify tải được). Sau khi đặt đúng `shape_predictor_68_face_landmarks.dat` (giải nén từ `.bz2`) vào `third_party/MaskTheFace/dlib_models/`: chạy `mask_the_face.py` thật trên 1 ảnh thật (2 người, nguồn: ảnh ví dụ có sẵn trong chính repo `davisking/dlib`, dùng cho mục đích test pipeline) → **phát hiện đúng 2 khuôn mặt, đeo mask surgical đúng góc nghiêng từng mặt**. Sau đó chạy tiếp qua **`process_identity()` thật của chính project này** (không phải fake runner trong `tests/unit/test_preprocessing.py`) — đúng hàm đã sửa bug `cwd` (mục A1 trong khung "8 bug thật" ở trên) — kết quả: 4/4 ảnh output đúng (2 ảnh gốc × 2 mask_type `surgical`/`N95`), xác nhận **fix bug cwd hoạt động đúng với tool thật**, không chỉ đúng với fake runner trong test.

**Kết luận cho bạn**: phần MaskTheFace/dlib giờ đã có đường chạy thật rõ ràng (xem hướng dẫn chi tiết trong file hướng dẫn chạy riêng). Phần còn chặn cứng, không có cách nào vượt qua từ sandbox này: **`backbone.pth` pretrained thật** và **RMFD gốc (525 identity)** — cả 2 chỉ phát hành qua Baidu Pan/OneDrive, bắt buộc bạn tự tải trên máy của bạn.
