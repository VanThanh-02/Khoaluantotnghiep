# -*- coding: utf-8 -*-
import os
import pydicom
import numpy as np
import cv2
from skimage.transform import resize


def clean_mask(binary_mask):
    """Giữ lại thành phần connected lớn nhất trong mask."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return binary_mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = np.argmax(areas) + 1
    cleaned = np.zeros_like(binary_mask)
    cleaned[labels == max_idx] = 255
    return cleaned


def resize_with_padding(image, target_size=224, background_value=0):
    """Resize giữ chi tiết bằng skimage, thêm padding. GIỮ dtype gốc, KHÔNG ép 16-bit."""
    h, w = image.shape
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = resize(
        image,
        (new_h, new_w),
        order=0,              # nearest-neighbor → không mờ
        preserve_range=True,  # giữ nguyên range gốc
        anti_aliasing=False
    ).astype(image.dtype)

    # Padding để ảnh vuông
    result = np.full((target_size, target_size), background_value, dtype=image.dtype)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return result


def crop_mammogram_dicom(input_path, output_folder, target_size=224):
    dcm = pydicom.dcmread(input_path)
    original_image = dcm.pixel_array     # GIỮ dtype gốc (8/12/14/16 bits), KHÔNG ép uint16

    # Xác định MONOCHROME1
    is_mono1 = getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1"

    # Ảnh 8-bit chỉ dùng để tạo mask (giữ y hệt logic ban đầu)
    arr = original_image.astype(np.float32, copy=False)
    lower, upper = np.percentile(arr, (1, 99))
    if upper <= lower:
        lower, upper = float(arr.min()), float(arr.max())
    clipped = np.clip(arr, lower, upper)
    img_8bit = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Otsu threshold (GIỮ nguyên hướng nhị phân như code gốc)
    _, mask = cv2.threshold(img_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_clean = clean_mask(mask)

    # Lấy contour lớn nhất (GIỮ logic ban đầu)
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_contour = np.zeros_like(mask_clean)
    if contours:
        cv2.drawContours(mask_contour, [max(contours, key=cv2.contourArea)], -1, 255, -1)

    coords = np.column_stack(np.where(mask_contour > 0))
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        cropped = original_image[y_min:y_max, x_min:x_max]
    else:
        cropped = original_image

    # Padding theo background (GIỮ nguyên cách chọn nền như code gốc)
    background_value = np.max(cropped) if is_mono1 else np.min(cropped)
    resized = resize_with_padding(cropped, target_size=target_size, background_value=background_value)

    # Lưu DICOM mới (GIỮ dtype gốc), đồng thời xử lý thẻ để tránh lỗi ghi
    cropped_dcm = dcm.copy()
    cropped_dcm.Rows, cropped_dcm.Columns = resized.shape
    cropped_dcm.PixelData = resized.astype(original_image.dtype).tobytes()

    # (Khuyến nghị) Xoá các thẻ dễ gây lỗi VR mơ hồ khi ghi
    # SmallestImagePixelValue (0028,0106), LargestImagePixelValue (0028,0107)
    for tag in [(0x0028, 0x0106), (0x0028, 0x0107)]:
        if tag in cropped_dcm:
            del cropped_dcm[tag]

    # Sau khi resize, PixelSpacing cũ không còn đúng → xoá (nếu bạn không tính lại)
    if "PixelSpacing" in cropped_dcm:
        del cropped_dcm.PixelSpacing

    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.basename(input_path).replace(".dicom", "_cropped.dcm").replace(".dcm", "_cropped.dcm")
    out_path = os.path.join(output_folder, base_name)

    # write_like_original=False để pydicom ghi an toàn
    pydicom.dcmwrite(out_path, cropped_dcm, write_like_original=False)
    print(f"✅ Đã lưu: {out_path}")


def process_folder(input_folder, output_folder, target_size=224):
    for file in os.listdir(input_folder):
        if file.lower().endswith((".dcm", ".dicom")):
            input_path = os.path.join(input_folder, file)
            try:
                crop_mammogram_dicom(input_path, output_folder, target_size=target_size)
            except Exception as e:
                print(f"⚠️ Lỗi {file}: {e}")


if __name__ == "__main__":
    # 👉 Chỉ cần sửa target_size ở đây
    target_size = 224

    input_folder = r"D:\DO AN TOT NGHIEP - UTV\IMAGE\Image-DICOM-goc\BVUB\4.Normal"
    output_folder = r"D:\DO AN TOT NGHIEP - UTV\IMAGE\Image-DICOM-TXL\IMAGE-224X224\BVUB-CUT-RESIZE-224X224\4.Normal"
    process_folder(input_folder, output_folder, target_size=target_size)
