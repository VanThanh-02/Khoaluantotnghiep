import os
import random
import shutil
from tqdm import tqdm

def process_folder(folder_path, output_folder, target_count):
    """
    Chọn ngẫu nhiên ảnh từ folder_path để đạt đủ target_count
    và lưu vào output_folder.
    """
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.dicom', '.dcm'))]
    current_count = len(images)

    # Nếu số lượng ảnh đã đủ hoặc lớn hơn target_count thì chỉ copy ngẫu nhiên target_count ảnh
    if current_count >= target_count:
        print(f"Thư mục {folder_path} đã có {current_count} ảnh. Chọn ngẫu nhiên {target_count} ảnh.")
        selected_images = random.sample(images, target_count)
    else:
        print(f"Thư mục {folder_path} chỉ có {current_count} ảnh. Sử dụng toàn bộ và nhân bản để đạt {target_count} ảnh.")
        selected_images = images.copy()
        while len(selected_images) < target_count:
            selected_images.append(random.choice(images))

    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(output_folder, exist_ok=True)

    # Sao chép ảnh được chọn vào thư mục đích
    for image_path in tqdm(selected_images, desc=f"Đang xử lý {folder_path}"):
        file_name = os.path.basename(image_path)
        shutil.copy(image_path, os.path.join(output_folder, file_name))

def main():
    input_dir = r"D:\DO AN TOT NGHIEP - UTV\IMAGE\Image-DICOM-TXL\IMAGE-224X224\VIN-CUT-RESIZE-224X224"
    output_dir = r"D:\DO AN TOT NGHIEP - UTV\TAP-ANH-TRUONG-HOP-3\TH_3.2_Image_Dicom_224x224"
    target_count = 7967

    # Tên thư mục cần xử lý
    folder_names = ["4.Normal"]

    for folder_name in folder_names:
        folder_path = os.path.join(input_dir, folder_name)
        output_folder = os.path.join(output_dir, folder_name)

        # Thực hiện chọn ngẫu nhiên ảnh từ thư mục và lưu vào thư mục đích
        process_folder(folder_path, output_folder, target_count)

if __name__ == "__main__":
    main()
