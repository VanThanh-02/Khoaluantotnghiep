import os
import shutil
import math
import pydicom
import numpy as np
import cv2
from tqdm import tqdm

# ============================================================
# HÀM CHUNG: đọc / ghi / xoay ảnh DICOM
# ============================================================
def load_dicom(file_path):
    try:
        dcm = pydicom.dcmread(file_path)
        return dcm.pixel_array
    except Exception as e:
        print(f"Lỗi đọc {file_path}: {e}")
        return None

def save_dicom(image, template_path, save_path):
    try:
        dcm = pydicom.dcmread(template_path)
        if dcm.pixel_array.dtype != image.dtype:
            image = image.astype(dcm.pixel_array.dtype)
        dcm.PixelData = image.tobytes()
        dcm.Rows, dcm.Columns = image.shape
        dcm.SOPInstanceUID = pydicom.uid.generate_uid()
        dcm.save_as(save_path)
    except Exception as e:
        print(f"Lỗi lưu {save_path}: {e}")

def _read_photometric(path):
    try:
        meta = pydicom.dcmread(path, stop_before_pixels=True)
        return getattr(meta, "PhotometricInterpretation", "MONOCHROME2")
    except Exception:
        return "MONOCHROME2"

def _list_dicom_files(folder):
    return [f for f in os.listdir(folder)
            if f.lower().endswith(('.dcm', '.dicom'))]

def _format_angle_suffix(angle):
    sign = "pos" if angle > 0 else "neg"
    mag = abs(angle)
    mag_str = f"{int(round(mag))}" if abs(mag - round(mag)) < 1e-9 else f"{mag:.6f}".rstrip("0").rstrip(".")
    return f"rot_{sign}_{mag_str}"

def _rotate_image(img, angle, photometric):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    if photometric == "MONOCHROME2":
        return cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
    else:
        return cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101
        )

def _copy_original_and_flip(original_path, output_folder):
    """Lưu ảnh gốc và ảnh lật ngang."""
    base_name, ext = os.path.splitext(os.path.basename(original_path))
    photometric = _read_photometric(original_path)

    # Copy ảnh gốc sang output
    dst_orig = os.path.join(output_folder, base_name + ext)
    if not os.path.exists(dst_orig):
        try:
            shutil.copy(original_path, dst_orig)
        except Exception as e:
            print(f"Lỗi copy gốc {original_path}: {e}")

    # Đọc & lưu ảnh lật ngang
    img = load_dicom(original_path)
    if img is None:
        return None, None, None, None
    flip_h = cv2.flip(img, 1)
    save_dicom(flip_h, original_path, os.path.join(output_folder, f"{base_name}_flip_h{ext}"))
    return img, photometric, base_name, ext

# ============================================================
# Helper số thực an toàn
# ============================================================
def _frange(start, stop, step, include_end=True, ndigits=6):
    if step == 0:
        raise ValueError("step phải khác 0")
    # Bảo đảm hướng step đúng với chiều (start->stop)
    if (stop - start) * step < 0:
        step = -step
    x = start
    out = []
    eps = 1e-12

    def _round(v):
        return round(float(v), ndigits)

    if step > 0:
        while x < stop - eps:
            out.append(_round(x))
            x += step
        if include_end and x <= stop + eps:
            out.append(_round(stop))
    else:
        while x > stop + eps:
            out.append(_round(x))
            x += step
        if include_end and x >= stop - eps:
            out.append(_round(stop))
    return out

# ============================================================
# MODE 1: GRID (hỗ trợ nhiều khoảng)
# ============================================================
def _build_angle_list_grid(
    min_angle=None, max_angle=None, angle_step=None,
    segments=None,                 # [(min1,max1,step1), (min2,max2,step2), ...]
    exclude_zero=True,
    deduplicate=True,
    keep_order=True
):
    """
    Cách dùng:
      - 1 khoảng (API cũ):
          _build_angle_list_grid(min_angle=-7, max_angle=7, angle_step=0.5)
      - Nhiều khoảng:
          _build_angle_list_grid(segments=[(1,10,1.0), (-5,-1,0.5)])
    """
    angles = []

    if segments is not None:
        # Nhiều khoảng, giữ nguyên thứ tự liệt kê (nếu keep_order=True)
        for (lo, hi, st) in segments:
            if st <= 0:
                raise ValueError(f"angle_step phải > 0 cho khoảng ({lo},{hi},{st})")
            if lo <= hi:
                arr = _frange(lo, hi, st, include_end=True, ndigits=6)
            else:
                arr = _frange(lo, hi, -st, include_end=True, ndigits=6)
            angles.extend(arr)
    else:
        # 1 khoảng: tạo dãy đều chuẩn (không ràng buộc min<0<max)
        if angle_step is None or min_angle is None or max_angle is None:
            raise ValueError("Thiếu tham số GRID: cần (min_angle, max_angle, angle_step) hoặc segments=[...].")
        if angle_step <= 0:
            raise ValueError("angle_step phải > 0")
        if min_angle <= max_angle:
            angles = _frange(min_angle, max_angle, angle_step, include_end=True, ndigits=6)
        else:
            angles = _frange(min_angle, max_angle, -angle_step, include_end=True, ndigits=6)

    # Loại 0 nếu cần
    if exclude_zero:
        eps = 1e-9
        angles = [a for a in angles if abs(a) > eps]

    # Loại trùng & chuẩn hóa
    if deduplicate:
        seen = set()
        uniq = []
        for a in angles:
            ra = round(float(a), 6)
            if ra not in seen:
                uniq.append(ra)
                seen.add(ra)
        angles = uniq
    else:
        angles = [round(float(a), 6) for a in angles]

    # Sắp xếp nếu muốn
    if not keep_order:
        angles.sort()

    return angles

def process_folder_grid(
    folder_path, output_folder,
    min_angle=None, max_angle=None, angle_step=None,
    segments=None,                 # ví dụ: [(1,10,1.0), (-5,-1,0.5)]
    exclude_zero=True,
    deduplicate=True,
    keep_order=True
):
    files = _list_dicom_files(folder_path)
    if not files:
        print("Không có DICOM trong thư mục.")
        return
    os.makedirs(output_folder, exist_ok=True)

    angles = _build_angle_list_grid(
        min_angle=min_angle, max_angle=max_angle, angle_step=angle_step,
        segments=segments,
        exclude_zero=exclude_zero,
        deduplicate=deduplicate,
        keep_order=keep_order
    )

    if segments is not None:
        print(f"[GRID] Nhiều khoảng: {segments} -> Tổng {len(angles)} góc")
    else:
        print(f"[GRID] 1 khoảng: [{min_angle}, {max_angle}], step={angle_step} -> Tổng {len(angles)} góc")

    for filename in tqdm(files, desc="GRID"):
        src = os.path.join(folder_path, filename)
        img, photometric, base_name, ext = _copy_original_and_flip(src, output_folder)
        if img is None:
            continue
        for angle in angles:
            rotated = _rotate_image(img, angle, photometric)
            save_dicom(rotated, src, os.path.join(
                output_folder, f"{base_name}_{_format_angle_suffix(angle)}{ext}"
            ))
    print("✅ Hoàn tất GRID.")

# ============================================================
# MODE 2: EXPLICIT
# ============================================================
def _normalize_angle_list_explicit(angles):
    out, seen = [], set()
    for a in angles or []:
        try:
            v = round(float(a), 6)
            if abs(v) < 1e-9:
                continue
            if v not in seen:
                out.append(v)
                seen.add(v)
        except:
            pass
    return out

def process_folder_explicit(folder_path, output_folder, angles_list):
    files = _list_dicom_files(folder_path)
    if not files:
        print("Không có DICOM trong thư mục.")
        return
    os.makedirs(output_folder, exist_ok=True)
    angles = _normalize_angle_list_explicit(angles_list)
    print(f"[EXPLICIT] Góc: {angles}")
    for filename in tqdm(files, desc="EXPLICIT"):
        src = os.path.join(folder_path, filename)
        img, photometric, base_name, ext = _copy_original_and_flip(src, output_folder)
        if img is None:
            continue
        for angle in angles:
            rotated = _rotate_image(img, angle, photometric)
            save_dicom(rotated, src, os.path.join(output_folder, f"{base_name}_{_format_angle_suffix(angle)}{ext}"))
    print("✅ Hoàn tất EXPLICIT.")

# ============================================================
# MODE 3: RANDOM (chỉ giữ target_total_images)
# ============================================================
def _sample_random_angles(min_angle, max_angle, k, rng,
                          exclude_zero=True, deduplicate=True):
    if k <= 0 or max_angle <= min_angle:
        return []
    eps = 1e-9
    res, seen = [], set()
    while len(res) < k:
        a = rng.uniform(min_angle, max_angle)
        if exclude_zero and abs(a) < eps:
            continue
        a = round(float(a), 6)
        if deduplicate and a in seen:
            continue
        res.append(a)
        seen.add(a)
    return res

def process_folder_random(folder_path, output_folder,
                          random_min_angle=-10, random_max_angle=10,
                          target_total_images=40,
                          random_seed=123,
                          random_deduplicate=True,
                          shuffle_files=True,
                          shuffle_seed=123):
    files = _list_dicom_files(folder_path)
    if not files:
        print("Không có DICOM trong thư mục.")
        return
    os.makedirs(output_folder, exist_ok=True)

    N = len(files)
    base_per_file = 2  # (1 gốc + 1 flip)
    rng = np.random.default_rng(random_seed)

    # Shuffle danh sách file để chia đều xoay
    if shuffle_files:
        print(f"🔀 Shuffle file theo seed={shuffle_seed}")
        rng_shuffle = np.random.default_rng(shuffle_seed)
        rng_shuffle.shuffle(files)

    # Tính số xoay mỗi ảnh
    base_total = N * base_per_file
    desired_total = int(target_total_images)
    if desired_total <= base_total:
        per_file_K = [0] * N
    else:
        extra = desired_total - base_total
        base_k = extra // N
        remainder = extra % N
        per_file_K = [(base_k + 1 if i < remainder else base_k) for i in range(N)]
    print(f"[RANDOM] Tổng ảnh mục tiêu: {target_total_images}")

    for idx, filename in enumerate(tqdm(files, desc="RANDOM")):
        src = os.path.join(folder_path, filename)
        img, photometric, base_name, ext = _copy_original_and_flip(src, output_folder)
        if img is None:
            continue
        K = per_file_K[idx]
        angles = _sample_random_angles(random_min_angle, random_max_angle, K, rng,
                                       exclude_zero=True, deduplicate=random_deduplicate)
        for a in angles:
            rotated = _rotate_image(img, a, photometric)
            save_dicom(rotated, src, os.path.join(output_folder, f"{base_name}_{_format_angle_suffix(a)}{ext}"))

    total_out = len(_list_dicom_files(output_folder))
    print(f"✅ Hoàn tất RANDOM. Tổng ảnh đầu ra: {total_out}")

# ============================================================
# MAIN — chọn mode cần chạy
# ============================================================
def main():
    # Sửa lại đường dẫn theo máy của bạn
    input_dir  = r"D:\DO AN TOT NGHIEP - UTV\TAP-ANH-TRUONG-HOP-3\TH_3.1_Image_Dicom_224x224"
    output_dir = r"D:\DO AN TOT NGHIEP - UTV\TAP-ANH-TRUONG-HOP-3\TH_3.4_Image_Dicom_224x224"
    folder_names = ["1.Mass", "2.Calcification", "3.Asymmetry-Architectural"]  # ví dụ
    # "2.Calcification", "3.Asymmetry-Architectural"

    MODE = "RANDOM"   # "GRID" / "EXPLICIT" / "RANDOM"

    for folder_name in folder_names:
        src = os.path.join(input_dir, folder_name)
        dst = os.path.join(output_dir, folder_name)
        os.makedirs(dst, exist_ok=True)

        if MODE == "GRID":
            # --- ví dụ NHIỀU KHOẢNG như yêu cầu
            # process_folder_grid(
            #     src, dst,
            #     segments=[(1, 10, 1.0), (-5, -1, 1.0)],
            #     exclude_zero=True,
            #     deduplicate=True,
            #     keep_order=True
            # )

            # cũng có thể dùng kiểu 1 khoảng (API cũ):
            process_folder_grid(src, dst, min_angle=-10, max_angle=10, angle_step=1.0)

        elif MODE == "EXPLICIT":
            process_folder_explicit(src, dst, angles_list=[-5, 5])

        elif MODE == "RANDOM":
            process_folder_random(
                folder_path=src,
                output_folder=dst,
                random_min_angle=-10,
                random_max_angle=10,
                target_total_images=20000,
                random_seed=124,
                random_deduplicate=True,
                shuffle_files=True,
                shuffle_seed=124
            )
        else:
            print("MODE không hợp lệ.")

if __name__ == "__main__":
    main()
