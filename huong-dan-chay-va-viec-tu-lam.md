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

> **Cập nhật 2026-10-06**: bản trước thiếu bước `dvc add` cho dữ liệu raw
> (RMFD/enrollment/backbone weight) và thiếu `dvc push` sau `dvc repro` — hậu
> quả là Colab `dvc pull` không kéo về được gì ngoài các file do pipeline tự
> sinh. Đã bổ sung mục 3.1 và 5.1, kèm xác nhận bằng thực nghiệm (dựng 1 repo
> DVC thu nhỏ để kiểm chứng hành vi `deps:` vs `outs:`, không suy đoán).
>
> **Cập nhật 2026-10-07**: phát hiện thêm `data/splits/*.csv` (output của
> `make_splits`) không nên đi qua DVC — nhỏ, nên để git track trực tiếp
> (`git diff` xem được nội dung đổi, không cần `dvc pull` chỉ để lấy vài file
> text). Đã **sửa thẳng `dvc.yaml`** (thêm `cache: false` cho 4 file CSV) và
> `notebooks/train_colab.ipynb`, không chỉ sửa guide — đã verify bằng thực
> nghiệm riêng (xác nhận `git clone` một mình, chưa chạy `dvc pull`, đã đọc
> được đúng nội dung CSV).

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

**Nếu chạy `pip-compile` trên Windows và vẫn đầy ổ C: dù đã trỏ `TEMP`/`TMP`/`PIP_CACHE_DIR` sang ổ khác** — đọc source `pip-tools` 7.6.1 xác nhận nguyên nhân: `pip-compile` có **cache riêng** (`piptools/locations.py`: `CACHE_DIR = user_cache_dir("pip-tools")`, mặc định Windows là `%LOCALAPPDATA%\pip-tools\Cache`, tức `C:\Users\<tên>\AppData\Local\pip-tools\Cache`), và với resolver mặc định (backtracking) nó **truyền thư mục đó cho pip qua `--cache-dir`** (`scripts/compile.py`) — đè lên `PIP_CACHE_DIR`/`pip config` của bạn. Wheel tải về để resolve/hash (`.../pkgs`) cũng nằm dưới đây. `TEMP`/`TMP` không ảnh hưởng vì đường dẫn này lấy từ `LOCALAPPDATA`. Cách sửa (cmd):

```bat
set PIP_TOOLS_CACHE_DIR=F:\pip-tools-cache
pip-compile --generate-hashes --no-header requirements/train.in -o requirements/train.txt
```

Cố định cho các terminal sau: `setx PIP_TOOLS_CACHE_DIR F:\pip-tools-cache`. Dọn phần đã chiếm trên C: `rmdir /s /q "%LOCALAPPDATA%\pip-tools"`. (Chưa chạy thử trên Windows thật — phần "pip-tools đè cache-dir" đã đọc từ source, còn việc lỗi ENOSPC trong log của bạn đúng là ghi vào thư mục này là suy luận khớp với traceback.)

**Cảnh báo — nên sinh `train.txt` trên Linux (Colab/WSL), không phải Windows**: `pip-compile` chỉ resolve cho đúng hệ điều hành + phiên bản Python đang chạy, và pip-tools ghi rõ output phụ thuộc nền tảng. `torch` trên Linux kéo thêm nhóm `nvidia-*`/`triton`; lock compile trên Windows sẽ không có các gói đó, còn CI (`ubuntu-latest`, `lock-sync` recompile rồi so) và Colab đều là Linux — nên rất dễ lệch/đỏ. Python trong venv của bạn (3.12) cũng lệch `PYTHON_VERSION` trong `ci.yml` (3.11). Cách gọn nhất: chạy 3 lệnh `pip-compile` ở trên **trên Colab** (kiểm `python --version`, rồi đặt `PYTHON_VERSION` trong `.github/workflows/ci.yml` cho khớp), tải `requirements/train.txt` về commit. (Phần "platform-specific" dựa trên tài liệu pip-tools, mình chưa tự kiểm bằng cách compile trên 2 hệ điều hành.)

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

### 3.1 Track bằng DVC — bước BẮT BUỘC, bản guide trước thiếu đúng chỗ này

**Đã verify bằng thực nghiệm riêng (2026-10-06, dựng 1 repo DVC thu nhỏ y hệt cấu trúc project này để kiểm chứng, không suy đoán từ tài liệu)**: chỉ đặt file vào `data/raw/...` rồi chạy `dvc repro` **KHÔNG đủ** để dữ liệu được `dvc push` lên remote. Lý do: `dvc.yaml` liệt kê `data/raw/rmfd/masked`/`unmasked` như `deps:` của stage `make_splits`/`augment_masks`, nhưng **3 thư mục đó không phải `outs:` của bất kỳ stage nào** — DVC chỉ tự động đưa **`outs:`** (output của stage, vd `data/processed/masktheface/rmfd`, `data/splits/*.csv`) vào cache để push. `deps:` là dữ liệu bên ngoài do bạn tự đặt vào, DVC chỉ hash để biết stage có cần chạy lại hay không, **KHÔNG tự cache/push nội dung** — thực nghiệm xác nhận: không `dvc add` thì `dvc push` chỉ đẩy đúng phần `outs:`, một `dvc pull` trên máy khác (Colab) sẽ KHÔNG kéo về được `data/raw/rmfd/masked`/`unmasked`/`enrollment` — đúng như bạn phát hiện.

**Cách đúng — chạy ngay sau khi đặt ảnh RMFD vào đúng cấu trúc ở trên, TRƯỚC `dvc repro`:**

```bash
dvc add data/raw/rmfd/masked data/raw/rmfd/unmasked data/raw/enrollment
git add data/raw/rmfd/masked.dvc data/raw/rmfd/unmasked.dvc data/raw/enrollment.dvc data/.gitignore
git commit -m "data: add RMFD raw + enrollment thật"
```

Lệnh `dvc add` tạo file `.dvc` (con trỏ nhỏ, chứa hash — commit vào git bình thường) cho từng thư mục — đây chính là phần guide trước thiếu ("không có tạo các file .dvc" đúng như bạn chỉ ra). Thiếu bước này thì `data/raw/...` không bao giờ vào được DVC cache, dù `dvc repro`/`dvc push` chạy bao nhiêu lần.

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

4. `dvc add models/pretrained/backbone.pth` (file nhị phân, không commit thẳng vào git). **Chưa cần `dvc push` ngay** — gộp chung 1 lần `dvc push` với dữ liệu RMFD ở mục 5.1 cho gọn (push nhiều lần cũng không sai, chỉ dư bước).

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

Hoặc dùng DVC để chạy cả 2 stage theo đúng dependency graph (`dvc.yaml`) — **khuyến nghị dùng cách này**, vì `dvc repro` tự ghi lại hash vào `dvc.lock`, cần thiết cho bước push ngay sau:

```bash
dvc repro
```

Kết quả mong đợi: `data/splits/{train,val,test,enrollment_demo}.csv` xuất hiện, log in ra số identity/ảnh mỗi split (xem `src/data/labels.py:run()` — log cả cảnh báo nếu có identity thiếu ảnh masked/unmasked thật).

`configs/data.yaml` có `max_images_per_identity: 5` (cap số ảnh/identity đưa vào MaskTheFace, tránh chạy hết ~270K job con ngay lần đầu) — tăng dần sau khi xác nhận pipeline chạy ổn với subset nhỏ.

### 5.1 Push lên DagsHub — bước BẮT BUỘC, nếu không Colab sẽ `dvc pull` về tay không

`dvc repro` chạy xong chỉ cập nhật **local cache** trên máy bạn — chưa có gì lên DagsHub cả. Đẩy lên remote + commit con trỏ vào git (thiếu `git push` thì Colab `git clone`/`git pull` cũng không biết gì để mà `dvc pull`):

```bash
dvc push
git add dvc.lock
git commit -m "data: chạy augment_masks + make_splits trên RMFD thật"
git push
```

Tới đây, `dvc push` đẩy đúng 2 nhóm nhị phân: `data/raw/rmfd/{masked,unmasked}` + `data/raw/enrollment` (đã `dvc add` ở mục 3.1), và `data/processed/masktheface/*` (outs của `augment_masks`, tự cache khi `dvc repro` chạy) — cộng `models/pretrained/backbone.pth` (đã `dvc add` ở mục 4).

**Riêng `data/splits/*.csv` KHÔNG qua DVC nữa** (`dvc.yaml` đã đặt `cache: false` cho 4 file này — đã verify bằng thực nghiệm riêng, xem comment trong chính `dvc.yaml`): CSV nhỏ, để **git track trực tiếp** thay vì đẩy vào DVC cache/remote như ảnh nhị phân lớn — vừa xem được `git diff` thật khi split đổi, vừa không cần `dvc pull` chỉ để lấy vài file text. `git commit dvc.lock` ở trên đã kèm theo đúng nội dung CSV (vì CSV giờ là file git thường, nằm trong cùng commit) — `git push` ở cuối đẩy cả 2.

Thiếu bất kỳ bước `dvc add`/`dvc push`/`git push` nào ở trên — máy khác (Colab, mục 6) sẽ `git clone`/`dvc pull` "thành công" nhưng không về đủ file, rồi `train.py` lỗi `FileNotFoundError` khi tìm ảnh theo `image_path` trong manifest CSV (xem `src/training/dataset.py:load_batch` — đọc trực tiếp từ `data/raw/...` và `data/processed/...`, cả 2 đều phải có mặt; CSV thì chỉ cần `git clone`/`git pull` là đủ, không cần đợi `dvc pull`).

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
#      Điền models/pretrained/SOURCE.yaml

# 3.1+4 (BẮT BUỘC, hay bị bỏ sót — xem mục 3.1): dvc add cho MỌI raw input
dvc add data/raw/rmfd/masked data/raw/rmfd/unmasked data/raw/enrollment models/pretrained/backbone.pth
git add data/raw/rmfd/masked.dvc data/raw/rmfd/unmasked.dvc data/raw/enrollment.dvc \
        models/pretrained/backbone.pth.dvc data/.gitignore models/pretrained/.gitignore
git commit -m "data: add RMFD raw + enrollment + backbone weight thật"

# 5. Data pipeline (dùng dvc repro để dvc.lock được ghi đúng, cần cho bước push)
python -m src.data.preprocessing --config configs/data.yaml --dry-run   # soát trước
dvc repro

# 5.1 (BẮT BUỘC — thiếu bước này thì Colab dvc pull về tay không)
dvc push
git add dvc.lock
git commit -m "data: chạy augment_masks + make_splits"
git push

# 6. Train
export MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
export MLFLOW_TRACKING_USERNAME=<user> MLFLOW_TRACKING_PASSWORD=<token>
python -m src.training.train --config configs/finetune_occlusion.yaml --resume-dir <drive_path>

# 7. Eval
python -m src.evaluation.run_eval --config configs/finetune_occlusion.yaml \
  --inference-config configs/inference.yaml --checkpoint-dir <drive_path>
cat reports/eval_metrics.json
```
