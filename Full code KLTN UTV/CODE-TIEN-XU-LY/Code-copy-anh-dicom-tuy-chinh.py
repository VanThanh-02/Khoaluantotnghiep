# -*- coding: utf-8 -*-
"""
Copy DICOM (.dcm/.dicom) không thay đổi nội dung:
- Tạo N bản sao cho mỗi file với hậu tố đặt sau tên gốc: abc_X.dcm
- X có thể là số (tự sinh 01, 02, ...) hoặc danh sách ký tự/chuỗi tùy ý (["A","B","C"]).
- Giữ nguyên thời gian & metadata file (shutil.copy2), KHÔNG đọc/ghi lại DICOM.
- Hỗ trợ quét đệ quy, bảo toàn cấu trúc thư mục nguồn.

Cấu hình nhanh ở mục "CẤU HÌNH".
"""

import os
import sys
import csv
import shutil
from typing import List, Iterable
from tqdm import tqdm

# ======================
# CẤU HÌNH
# ======================
INPUT_DIR       = r"D:\DO AN TOT NGHIEP - UTV\IMAGE\Image-DICOM-TXL\IMAGE-224X224\MIAS-CUT-RESIZE-224X224\4.Normal"       # Thư mục nguồn (A)
OUTPUT_DIR      = r"D:\DO AN TOT NGHIEP - UTV\TAP-ANH-TRUONG-HOP-3\TH_3.2_Folder-4.Normal\4.Normal-MIAS"       # Thư mục đích (B)
RECURSIVE       = True                  # Quét đệ quy thư mục con
PRESERVE_TREE   = True                  # Giữ cấu trúc thư mục con
COPY_ORIGINAL   = False                 # Sao chép thêm 1 bản tên gốc (không hậu tố)

# Số bản sao / file (nếu SUFFIX_TOKENS rỗng, sẽ tự sinh số 01..NN)
N_COPIES        = 1

# Danh sách hậu tố tùy chỉnh; nếu KHÔNG rỗng, số bản sao = min(N_COPIES, len(SUFFIX_TOKENS))
# Ví dụ: ["A","B","C"] hoặc ["x1","x2","x3"], hoặc list("ABC")
SUFFIX_TOKENS: List[str] = [4]

# Nếu SUFFIX_TOKENS rỗng -> sinh số:
START_AT        = None     # bắt đầu từ số mấy
PAD_WIDTH       = None     # độ rộng zero-pad, ví dụ 2 -> 01, 02, ...
SEPARATOR       = "_"   # dấu nối giữa tên gốc và hậu tố

# Chính sách khi đụng trùng tên ở đích:
#   "skip"      -> bỏ qua
#   "overwrite" -> ghi đè
#   "rename"    -> tự tăng thêm __1, __2, ... cho đến khi không trùng
ON_CONFLICT     = "skip"

# Chỉ copy các phần mở rộng sau (không phân biệt hoa thường)
VALID_EXTS      = {".dcm", ".dicom"}

# Ghi manifest ánh xạ src->dst
WRITE_MANIFEST  = True
MANIFEST_PATH   = os.path.join(OUTPUT_DIR, "copy_manifest.csv")

# Chạy thử (không ghi file)
DRY_RUN         = False
# ======================


def is_dicom_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in VALID_EXTS


def iter_source_files(root: str, recursive: bool) -> Iterable[str]:
    if not recursive:
        for name in os.listdir(root):
            p = os.path.join(root, name)
            if os.path.isfile(p) and is_dicom_file(p):
                yield p
    else:
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                p = os.path.join(dirpath, name)
                if os.path.isfile(p) and is_dicom_file(p):
                    yield p


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def gen_tokens(n: int) -> List[str]:
    """Sinh danh sách token hậu tố cho 1 file."""
    if SUFFIX_TOKENS:
        return SUFFIX_TOKENS[:n]
    return [str(i).zfill(PAD_WIDTH) for i in range(START_AT, START_AT + n)]


def make_target_path(src_file: str, base_outdir: str, keep_tree: bool) -> str:
    """Tính thư mục đích của file (chỉ thư mục), có thể bảo toàn cấu trúc."""
    if not keep_tree:
        return base_outdir
    rel = os.path.relpath(os.path.dirname(src_file), start=INPUT_DIR)
    # Nếu src_file không nằm trong INPUT_DIR (rel bắt đầu bằng ..), vẫn đổ về root OUTPUT_DIR
    if rel.startswith(".."):
        rel = ""
    return os.path.join(base_outdir, rel)


def unique_path(desired_path: str) -> str:
    """Nếu ON_CONFLICT='rename', tạo tên duy nhất bằng cách thêm __1, __2..."""
    if not os.path.exists(desired_path):
        return desired_path
    stem, ext = os.path.splitext(desired_path)
    k = 1
    while True:
        candidate = f"{stem}__{k}{ext}"
        if not os.path.exists(candidate):
            return candidate
        k += 1


def plan_copies_for_file(src: str) -> List[str]:
    """Lập danh sách đường dẫn đích (đầy đủ) sẽ tạo cho 1 file nguồn."""
    dst_dir = make_target_path(src, OUTPUT_DIR, PRESERVE_TREE)
    ensure_dir(dst_dir)

    src_name = os.path.basename(src)
    stem, ext = os.path.splitext(src_name)

    targets = []

    # 1) (tuỳ chọn) Bản copy giữ nguyên tên gốc
    if COPY_ORIGINAL:
        base_path = os.path.join(dst_dir, f"{stem}{ext}")
        targets.append(base_path)

    # 2) Bản copy với hậu tố
    tokens = gen_tokens(N_COPIES)
    for t in tokens:
        out_name = f"{stem}{SEPARATOR}{t}{ext}"
        targets.append(os.path.join(dst_dir, out_name))

    return targets


def resolve_conflict(path: str) -> str:
    if not os.path.exists(path):
        return path
    if ON_CONFLICT == "skip":
        return ""   # báo hiệu bỏ qua
    if ON_CONFLICT == "overwrite":
        return path
    if ON_CONFLICT == "rename":
        return unique_path(path)
    # Mặc định an toàn
    return ""


def copy_one(src: str, dst: str) -> bool:
    if DRY_RUN:
        return True
    try:
        # copy2 giữ nguyên mtime/atime/permission; nội dung file không đổi
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        print(f"[ERR] Không thể copy: {src} -> {dst} | {e}")
        return False


def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"[ERR] INPUT_DIR không tồn tại: {INPUT_DIR}")
        sys.exit(1)
    ensure_dir(OUTPUT_DIR)

    src_files = list(iter_source_files(INPUT_DIR, RECURSIVE))
    if not src_files:
        print("[INFO] Không tìm thấy file DICOM nào.")
        return

    manifest_rows = []
    total_planned = 0
    total_done = 0
    total_skipped = 0
    total_conflict_renamed = 0

    print(f"[INFO] Tìm thấy {len(src_files)} file nguồn. Bắt đầu copy...")
    for src in tqdm(src_files, ncols=80):
        targets = plan_copies_for_file(src)
        for desired in targets:
            total_planned += 1
            resolved = resolve_conflict(desired)
            if not resolved:
                total_skipped += 1
                continue
            if resolved != desired and ON_CONFLICT == "rename":
                total_conflict_renamed += 1

            ok = copy_one(src, resolved)
            if ok:
                total_done += 1
                manifest_rows.append((src, resolved))
            else:
                total_skipped += 1

    # Ghi manifest
    if WRITE_MANIFEST:
        try:
            ensure_dir(os.path.dirname(MANIFEST_PATH))
            with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["source", "destination"])
                w.writerows(manifest_rows)
        except Exception as e:
            print(f"[WARN] Không thể ghi manifest: {MANIFEST_PATH} | {e}")

    # Tóm tắt
    print("\n===== TÓM TẮT =====")
    print(f"File nguồn             : {len(src_files)}")
    print(f"Kế hoạch bản sao       : {total_planned}")
    print(f"Đã copy thành công     : {total_done}")
    print(f"Bỏ qua / lỗi           : {total_skipped}")
    if ON_CONFLICT == "rename":
        print(f"Đã tự đổi tên để tránh trùng: {total_conflict_renamed}")
    print(f"Manifest                : {MANIFEST_PATH if WRITE_MANIFEST else '(tắt)'}")
    if DRY_RUN:
        print("[DRY-RUN] Không có file nào được ghi (chạy thử).")


if __name__ == "__main__":
    main()
