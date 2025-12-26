import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import os, cv2, pydicom, numpy as np
from CODE_CUT_RESIZE_CHUAN import crop_mammogram_dicom
from Code_du_doan_ResNet50_tfdata_dicom import predict_image

# ====== GIAO DIỆN ======
root = tk.Tk()
root.title("🩻 Giao diện xử lý & phân loại ảnh DICOM")
root.geometry("1300x700")
root.configure(bg="#2b2b2b")

Label(root, text="PHÂN TÍCH ẢNH DICOM", font=("Arial", 20, "bold"), fg="white", bg="#2b2b2b").pack(pady=10)

frame_images = tk.Frame(root, bg="#2b2b2b")
frame_images.pack(pady=20)

# Khung 3 ảnh
frames = {}
labels = {}
for name in ["Ảnh gốc", "Ảnh sau tiền xử lý", "Ảnh đưa vào mô hình"]:
    f = tk.Frame(frame_images, bg="#2b2b2b")
    f.pack(side="left", padx=10)
    Label(f, text=name, fg="white", bg="#2b2b2b", font=("Arial", 14, "bold")).pack()
    lbl = Label(f, bg="#2b2b2b")
    lbl.pack(padx=5, pady=5)
    frames[name] = f
    labels[name] = lbl

result_text = tk.StringVar()
Label(root, textvariable=result_text, fg="yellow", bg="#2b2b2b", font=("Arial", 16)).pack(pady=10)


def show_image(arr, panel):
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    img = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
    img = Image.fromarray(np.uint8(img)).resize((350, 350))
    img_tk = ImageTk.PhotoImage(img)
    panel.configure(image=img_tk)
    panel.image = img_tk


def select_image():
    path = filedialog.askopenfilename(filetypes=[("DICOM files", "*.dcm;*.dicom")])
    if not path:
        return
    result_text.set("Đang xử lý ảnh...")
    root.update()

    # 1️⃣ Hiển thị ảnh gốc
    dcm = pydicom.dcmread(path)
    img_orig = dcm.pixel_array
    show_image(img_orig, labels["Ảnh gốc"])

    # 2️⃣ Tiền xử lý
    temp_dir = "temp_processed"
    os.makedirs(temp_dir, exist_ok=True)
    proc_path = crop_mammogram_dicom(path, temp_dir, target_size=224)
    dcm_proc = pydicom.dcmread(proc_path)
    img_proc = dcm_proc.pixel_array
    show_image(img_proc, labels["Ảnh sau tiền xử lý"])

    # 3️⃣ Chuẩn bị ảnh model
    img_rgb = cv2.cvtColor(img_proc.astype(np.float32), cv2.COLOR_GRAY2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224))
    show_image(img_rgb, labels["Ảnh đưa vào mô hình"])

    # 4️⃣ Dự đoán
    cls, conf = predict_image(img_rgb)
    result_text.set(f"Kết quả: {cls}\nĐộ tin cậy: {conf:.2f}%")


Button(root, text="📂 Chọn ảnh DICOM", command=select_image,
       bg="#4CAF50", fg="white", font=("Arial", 14, "bold")).pack(pady=15)

root.mainloop()
