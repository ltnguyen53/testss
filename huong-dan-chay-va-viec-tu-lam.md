# Hướng dẫn chạy project — từ bootstrap tới có kết quả eval thật

> Phạm vi thực tế của guide này: đưa bạn từ repo trống tới có **model đã train
> + report eval thật** (Phase 0 → Phase 3). **Guide này KHÔNG có bước "chạy
> demo API"** — xem mục cuối "Khoảng cách tới demo API" để biết chính xác vì
> sao và cần làm gì trước khi tới đó. Mọi lệnh dưới đây lấy đúng từ chữ ký CLI
> thật trong code (`argparse` trong `train.py`/`run_eval.py`/`preprocessing.py`/
> `labels.py`), không suy đoán. Phần nào Claude đã tự chạy thật và verify được
> trong sandbox viết code (không có weight/dataset gốc) sẽ ghi rõ "**đã
> verify**"; phần nào chỉ đọc code suy ra, chưa chạy thật, ghi rõ "**chưa
> verify, cần bạn tự xác nhận**".

---

## 0. Yêu cầu hệ thống

- Python 3.11 (khớp `PYTHON_VERSION` trong `.github/workflows/ci.yml` — lệch version thì lock file hash không khớp).
- Git, `pip`.
- Để train thật: máy có GPU (khuyến nghị Google Colab T4 free, theo đúng thiết kế project) hoặc CPU (chạy được nhưng chậm — code không ép buộc CUDA, `train.py` tự chọn `torch.device("cuda" if torch.cuda.is_available() else "cpu")`).
- Tài khoản DagsHub (DVC remote + MLflow server) — xem mục 3.

---

## 1. Bootstrap repo

```bash
git clone <repo-url> face-tracking-mlops
cd face-tracking-mlops
python -m venv .venv && source .venv/bin/activate   # hoặc conda, tuỳ bạn
pip install -e .
```

### 1.1 Sinh lock file (nếu `requirements/train.txt` chưa có trong repo)

```bash
pip install pip-tools
pip-compile --generate-hashes --no-header requirements/api.in -o requirements/api.txt
pip-compile --generate-hashes --no-header requirements/train.in -o requirements/train.txt
pip install -r requirements/train.txt
```

**Đã verify (2026-10-02)**: `requirements/api.txt` sinh được, idempotent (chạy lại ra file giống hệt). `requirements/train.txt` **chưa sinh được trong sandbox viết code** — `pip-compile` cho `torch`/`torchvision` cần tải wheel thật để resolve (không chỉ metadata như gói nhẹ), sandbox hết đĩa giữa chừng. Chạy lệnh trên trên máy có đủ đĩa (vài GB trống) — khuyến nghị chạy thẳng trên Colab, vừa đủ đĩa vừa đúng Python version Colab dùng để train thật sau này.

### 1.2 Pre-commit

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Lần đầu nhiều khả năng `ruff format` tự sửa vài file — `git add` phần đã sửa rồi chạy lại cho sạch.

### 1.3 Git + DVC + DagsHub

```bash
git init   # nếu chưa có .git
dvc init
dvc remote add -d dagshub-storage https://dagshub.com/<user>/<repo>.dvc
dvc remote modify dagshub-storage --local access_key_id <dagshub_token>
dvc remote modify dagshub-storage --local secret_access_key <dagshub_token>
dvc push   # push rỗng lần đầu để xác nhận remote hoạt động
```

Lấy `<dagshub_token>` từ DagsHub → Settings → Tokens.

### 1.4 Secrets

Copy `.env.example` → `.env` (root, dùng cho API sau này) và điền `MLFLOW_TRACKING_URI`, `DAGSHUB_TOKEN`. Cho training (`train.py`/`run_eval.py` gọi `mlflow` trực tiếp), set qua biến môi trường trước khi chạy (Colab: Colab Secrets hoặc cell đầu notebook):

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
export MLFLOW_TRACKING_USERNAME=<dagshub_username>
export MLFLOW_TRACKING_PASSWORD=<dagshub_token>
```

`mlflow` tự đọc 3 biến này, không cần set gì thêm trong code.

---

## 2. MaskTheFace + dlib thật

### 2.1 Thêm submodule

```bash
git submodule add https://github.com/aqeelanwar/MaskTheFace.git third_party/MaskTheFace
git submodule update --init --recursive
```

### 2.2 Cài dependency — ĐÃ VERIFY cách làm dưới đây chạy thật (2026-10-03)

`third_party/MaskTheFace/requirements.txt` pin `dlib==19.19.0` (bản 2020) — PyPI chính thức của `dlib` **không có wheel dựng sẵn** cho Python hiện đại (chỉ có source distribution, cần compile bằng cmake/g++, tốn thời gian + disk không lường trước). Dùng gói thay thế này **thay vì** dòng `dlib==...` trong file đó:

```bash
pip install dlib-bin   # bên thứ 3, có wheel sẵn — đã verify: cài xong `import dlib` chạy ngay, không cần compile
pip install dotmap imutils opencv-python-headless tqdm requests Pillow numpy   # phần còn lại của requirements.txt, đều nhẹ
```

(`dlib-bin` verify trên Linux cp312 — 4.2MB, cài trong vài giây. Nếu platform/Python version của bạn không có wheel sẵn cho `dlib-bin`, xem trang PyPI của gói này để biết platform nào được hỗ trợ.)

### 2.3 Dlib shape predictor model (~99MB)

`MaskTheFace` có hàm `download_dlib_model()` tự tải model này nếu chưa có — **NHƯNG đã verify hàm đó tải từ `http://dlib.net/...`, một domain không phải lúc nào cũng reach được** (sandbox viết code dự án này không reach được domain đó dù reach GitHub/PyPI bình thường — tải về 0 byte, lỗi `bz2.OSError: Invalid data stream` lúc giải nén). Trên Colab (internet không giới hạn domain) nhiều khả năng tự tải được — cứ thử chạy bước 2.4 trước; nếu lỗi đúng kiểu trên, tải tay qua mirror GitHub (**đã verify cách này tải được và model chạy đúng**):

```bash
mkdir -p third_party/MaskTheFace/dlib_models
curl -L -o third_party/MaskTheFace/dlib_models/shape_predictor_68_face_landmarks.dat.bz2 \
  https://raw.githubusercontent.com/davisking/dlib-models/master/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d third_party/MaskTheFace/dlib_models/shape_predictor_68_face_landmarks.dat.bz2
```

### 2.4 Kiểm tra MaskTheFace chạy được — khuyến nghị làm TRƯỚC khi đụng vào RMFD thật

```bash
cd third_party/MaskTheFace
python mask_the_face.py --path <thư_mục_chứa_1_ảnh_mặt_thật> --mask_type surgical --verbose
```

**Đã verify thật (2026-10-03)**: với 1 ảnh thật có 2 khuôn mặt, lệnh trên in `Faces found: 2` và tạo đúng `<thư_mục>_masked/<tên_ảnh>_surgical.jpg` với mask đeo đúng góc nghiêng từng mặt. Nếu bạn thấy kết quả tương tự (không phải `Faces found: 0` hay crash) — môi trường đã sẵn sàng cho bước 4.

---

## 3. Dữ liệu RMFD thật — **việc bạn PHẢI tự làm, không có cách nào khác**

Đã thử tìm mọi kênh reach được từ sandbox viết code (GitHub, PyPI) — **không có cách nào lấy được bản RMFD gốc (525 identity, paired masked+unmasked) ngoài Baidu Pan**:

1. Vào trang GitHub chính thức của dataset: `X-zhangyang/Real-World-Masked-Face-Dataset` — đọc README, lấy link Baidu Pan.
2. Tải về (cần tài khoản Baidu, có thể cần VPN tuỳ khu vực của bạn).
3. Giải nén, sắp xếp đúng cấu trúc `configs/data.yaml` mong đợi:

```
data/raw/rmfd/masked/<identity_id>/*.jpg      # RMFRD thật, is_masked=true
data/raw/rmfd/unmasked/<identity_id>/*.jpg    # RMFRD thật, is_masked=false — nguồn để MaskTheFace augment thêm
data/raw/enrollment/<identity_id>/*.jpg       # ảnh cá nhân bạn tự chụp (demo), KHÔNG vào train/val/test
```

`<identity_id>` phải **khớp nhau** giữa thư mục `masked/` và `unmasked/` cho cùng 1 người — đây là giả định cốt lõi của `src/data/labels.py` (xem `log_identity_composition()`, sẽ log cảnh báo nếu lệch).

**Lưu ý chất lượng dữ liệu**: nếu bạn thử mirror GitHub không chính thức thay vì Baidu Pan — đã tự kiểm tra bằng mắt 1 mirror cụ thể (`X-zhangyang/Real-World-Masked-Face-Dataset`, nhánh `RWMFD_part_1`) và phát hiện **một số thư mục "identity" lẫn ảnh quảng cáo/ảnh đám đông, không phải ảnh nhất quán 1 người thật**. Đừng tin mù quáng bất kỳ mirror nào — spot-check vài thư mục bằng mắt trước khi train thật.

---

## 4. Backbone weight thật — **việc bạn PHẢI tự làm**

Cũng chỉ phát hành qua Baidu Pan/OneDrive, không có mirror GitHub nào tìm được:

1. Vào `deepinsight/insightface` → `recognition/arcface_torch` → README → bảng model zoo → chọn dòng khớp `backbone.arch` trong `configs/finetune_occlusion.yaml` (mặc định `r50`, train trên `MS1MV3`).
2. Tải `backbone.pth` (hoặc tên file tương tự trong model zoo) về `models/pretrained/backbone.pth`.
3. Điền `models/pretrained/SOURCE.yaml` (toàn bộ field, không được để `null` — `src/training/weights.py` raise nếu thiếu):

```bash
sha256sum models/pretrained/backbone.pth   # điền vào field sha256
```

Các field cần điền: `model_name`, `download_date`, `sha256`, `upstream_commit` (`git rev-parse HEAD` trong `third_party/insightface` nếu bạn clone kèm source, hoặc ghi commit tương ứng thời điểm tải). `architecture` đã sẵn `r50`, `upstream_repo`/`license_note` đã điền sẵn.

4. `dvc add models/pretrained/backbone.pth` (file nhị phân, không commit thẳng vào git) rồi `dvc push`.

**Giấy phép**: model zoo này chỉ dùng cho mục đích nghiên cứu phi thương mại (đã ghi sẵn trong `SOURCE.yaml`) — portfolio/demo ổn, không được định vị là sản phẩm thương mại.

---

## 5. Chạy pipeline data

```bash
# Dry-run trước — chỉ in lệnh MaskTheFace sẽ chạy, không thực thi, soát trước khi chạy thật
python -m src.data.preprocessing --config configs/data.yaml --dry-run

# Chạy thật — tạo ảnh masked synthetic từ data/raw/rmfd/unmasked (và enrollment)
python -m src.data.preprocessing --config configs/data.yaml

# Chia split identity-disjoint (train/val/test/enrollment_demo)
python -m src.data.labels --config configs/data.yaml
```

Hoặc dùng DVC để chạy cả 2 stage theo đúng dependency graph (`dvc.yaml`):

```bash
dvc repro
```

Kết quả mong đợi: `data/splits/{train,val,test,enrollment_demo}.csv` xuất hiện, log in ra số identity/ảnh mỗi split (xem `src/data/labels.py:run()` — log cả cảnh báo nếu có identity thiếu ảnh masked/unmasked thật).

`configs/data.yaml` có `max_images_per_identity: 5` (cap số ảnh/identity đưa vào MaskTheFace, tránh chạy hết ~270K job con ngay lần đầu) — tăng dần sau khi xác nhận pipeline chạy ổn với subset nhỏ.

---

## 6. Train thật

```bash
python -m src.training.train \
  --config configs/finetune_occlusion.yaml \
  --export-config configs/export.yaml \
  --resume-dir /content/drive/MyDrive/face-tracking-mlops/checkpoints
```

(`--resume-dir` override `checkpoint.dir` trong config — dùng đúng path Drive thật trên Colab. Bỏ flag này thì dùng path trong YAML.)

Nếu Colab ngắt giữa chừng: chạy LẠI đúng lệnh trên — `CheckpointManager` tự tìm checkpoint khớp `config_hash` trong `--resume-dir` và resume đúng epoch/step, không train lại từ đầu (đã verify bằng test tích hợp thật, xem `tests/unit/test_train.py`).

Theo dõi tiến trình qua MLflow (link DagsHub → Experiments → tên experiment trong `configs/finetune_occlusion.yaml:mlflow.experiment_name`).

---

## 7. Eval thật

```bash
python -m src.evaluation.run_eval \
  --config configs/finetune_occlusion.yaml \
  --inference-config configs/inference.yaml \
  --export-config configs/export.yaml \
  --checkpoint-dir /content/drive/MyDrive/face-tracking-mlops/checkpoints
```

Chạy xong sẽ:
- Ghi `reports/eval_metrics.json` — TAR@FAR theo từng nhóm occlusion (`overall`, `unmasked_unmasked`, `masked_masked`, `masked_unmasked`), cả val lẫn test.
- Cập nhật `configs/inference.yaml` — điền `threshold.value` (ngưỡng đã khoá) + `threshold.swept_on_config_hash`.
- Log MLflow run mới (metric `test_overall_tar` — registry đọc lại đúng tên này).
- Nếu metric tốt hơn Production hiện tại (hoặc chưa có Production nào) → tự đăng ký model version mới vào MLflow Model Registry, stage mặc định (**KHÔNG tự promote Production** — bạn tự xem lại rồi chuyển Staging → Production trên DagsHub UI).

Xem kết quả:
```bash
cat reports/eval_metrics.json | python -m json.tool
```

---

## Khoảng cách tới "demo API" — vì sao chưa có, cần làm gì trước

**Repo hiện tại KHÔNG CÓ API server nào chạy được** — đây không phải thiếu sót ngoài ý muốn, mà đúng tiến độ đã thống nhất (Phase 4 "Export + Client" và Phase 5 "Server + CI/CD" chưa bắt đầu code). Cụ thể, kiểm tra trực tiếp:

```bash
find src/api -type f        # chỉ có __init__.py rỗng
ls Dockerfile                # không tồn tại
find frontend -type f        # chỉ có .env.example + .gitkeep
```

Để có thứ gì đó demo được qua API, cần làm (theo đúng thứ tự phụ thuộc):

1. **`src/export/`** (Phase 4) — export model từ Phase 6 (checkpoint Production) sang ONNX, quantize, validate lại TAR@FAR sau quantize. Chưa có dòng code nào.
2. **`frontend/`** (Phase 4) — detect+landmark (RetinaFace/BlazeFace) + tracker (SORT/ByteTrack) + recognition (ONNX Runtime Web) chạy trên browser. Chưa có dòng code nào.
3. **`src/api/`** (Phase 5) — FastAPI thật: `/enrollment`, `/events`, `/health`, `/model-manifest`. Chỉ có `__init__.py` rỗng.
4. **`Dockerfile`** (Phase 5) — chưa tồn tại.

Không có cách nào "demo API" mà bỏ qua 4 mục trên — chúng chưa tồn tại dưới dạng code, không phải vấn đề cấu hình/chạy lệnh. Nếu bạn muốn đi tới demo API, bước tiếp theo hợp lý là implement Phase 4 trước (export cần model đã train xong — đúng những gì guide này vừa đưa bạn tới).

---

## Tổng hợp lệnh theo đúng thứ tự (checklist nhanh)

```bash
# 1. Bootstrap
git clone <repo-url> && cd face-tracking-mlops
pip install -e .
pip install pip-tools && pip-compile --generate-hashes --no-header requirements/train.in -o requirements/train.txt
pip install -r requirements/train.txt
pre-commit install && pre-commit run --all-files
dvc init && dvc remote add -d dagshub-storage https://dagshub.com/<user>/<repo>.dvc
dvc remote modify dagshub-storage --local access_key_id <token>
dvc remote modify dagshub-storage --local secret_access_key <token>

# 2. MaskTheFace + dlib
git submodule add https://github.com/aqeelanwar/MaskTheFace.git third_party/MaskTheFace
pip install dlib-bin dotmap imutils opencv-python-headless tqdm requests
mkdir -p third_party/MaskTheFace/dlib_models
curl -L -o third_party/MaskTheFace/dlib_models/shape_predictor_68_face_landmarks.dat.bz2 \
  https://raw.githubusercontent.com/davisking/dlib-models/master/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d third_party/MaskTheFace/dlib_models/shape_predictor_68_face_landmarks.dat.bz2

# 3+4. Tự tải tay: RMFD (Baidu Pan) -> data/raw/rmfd/{masked,unmasked}/<id>/
#                  backbone.pth (Baidu/OneDrive) -> models/pretrained/backbone.pth
#      Điền models/pretrained/SOURCE.yaml, rồi: dvc add models/pretrained/backbone.pth

# 5. Data pipeline
python -m src.data.preprocessing --config configs/data.yaml --dry-run
python -m src.data.preprocessing --config configs/data.yaml
python -m src.data.labels --config configs/data.yaml

# 6. Train
export MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
export MLFLOW_TRACKING_USERNAME=<user> MLFLOW_TRACKING_PASSWORD=<token>
python -m src.training.train --config configs/finetune_occlusion.yaml --resume-dir <drive_path>

# 7. Eval
python -m src.evaluation.run_eval --config configs/finetune_occlusion.yaml \
  --inference-config configs/inference.yaml --checkpoint-dir <drive_path>
cat reports/eval_metrics.json
```
