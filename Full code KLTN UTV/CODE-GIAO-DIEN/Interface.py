import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import os, cv2, pydicom, numpy as np

from CODE_CUT_RESIZE_CHUAN import crop_mammogram_dicom
from Code_du_doan_ResNet50_tfdata_dicom import predict_image


# ====================== CỬA SỔ CHÍNH ======================
root = tk.Tk()
root.title("ĐỒ ÁN TỐT NGHIỆP - PHÂN LOẠI UNG THƯ VÚ RESNET-50")
root.geometry("1420x920")
root.configure(bg="#06585E")


# ====================== HEADER ======================
header_frame = tk.Frame(root, bg="#A5F2E8", height=110)
header_frame.pack(fill="x", side="top")

# ---- TRÁI ----
left_frame = tk.Frame(header_frame, bg="#A5F2E8")
left_frame.pack(side="left", padx=20)

logo_left = Image.open(
    r"D:\DO AN TOT NGHIEP - UTV\ResNet-50-tfdata\CODE_INTERFACE\LOGO_UTE.png"
).resize((110, 90))
logo_left_tk = ImageTk.PhotoImage(logo_left)
Label(left_frame, image=logo_left_tk, bg="#A5F2E8").pack(side="left", padx=10)

Label(
    left_frame,
    text=(
        "TRƯỜNG ĐH SƯ PHẠM KỸ THUẬT TP.HCM\n"
        "BỘ MÔN ĐIỆN TỬ CÔNG NGHIỆP - Y SINH"
    ),
    font=("Arial", 18, "bold"),
    bg="#A5F2E8",
).pack(side="left")

# ---- PHẢI ----
right_frame = tk.Frame(header_frame, bg="#A5F2E8")
right_frame.pack(side="right", padx=20)

logo_right = Image.open(
    r"D:\DO AN TOT NGHIEP - UTV\ResNet-50-tfdata\CODE_INTERFACE\LOGO_KTYS.png"
).resize((110, 90))
logo_right_tk = ImageTk.PhotoImage(logo_right)
Label(right_frame, image=logo_right_tk, bg="#A5F2E8").pack(side="right", padx=10)

Label(
    right_frame,
    text=(
        "ĐỒ ÁN TỐT NGHIỆP\n"
        "PHÂN LOẠI UNG THƯ VÚ BẰNG RESNET-50\n"
        "GVHD: TS. Nguyễn Thanh Nghĩa\n"
        "SVTH: Phạm Thị Tâm Như - 21129078 | Trần Văn Thành - 21129090"
    ),
    font=("Arial", 13, "bold"),
    bg="#A5F2E8",
    justify="center",
).pack(side="right")


# ====================== KHUNG ẢNH ======================
main_frame = tk.Frame(root, bg="#06585E")
main_frame.pack_forget()

labels = {}
titles = ["Ảnh gốc", "Ảnh tiền xử lý"]

for i, name in enumerate(titles):
    f = tk.Frame(main_frame, bg="#004F53", width=480, height=480)
    f.grid(row=0, column=i, padx=110)
    f.pack_propagate(False)

    lbl = Label(f, bg="#004F53")
    lbl.pack(expand=True)
    labels[name] = lbl

    Label(
        main_frame,
        text=name,
        fg="#FFD966",
        bg="#06585E",
        font=("Consolas", 20, "bold"),
    ).grid(row=1, column=i, pady=(10, 0))


# ====================== KHỐI KẾT QUẢ ======================
result_title = Label(
    root,
    text="KẾT QUẢ PHÂN LOẠI",
    fg="black",
    bg="#06585E",
    font=("Arial", 22, "bold"),
)
result_title.place_forget()

result_class = Label(
    root,
    text="",
    fg="#FFD966",
    bg="#06585E",
    font=("Consolas", 28, "bold"),
)
result_class.place_forget()

result_conf = Label(
    root,
    text="",
    fg="white",
    bg="#06585E",
    font=("Consolas", 24, "bold"),
)
result_conf.place_forget()


# ====================== HIỂN THỊ ẢNH ======================
def show_image(arr, panel, target_size=480):

    # Màu nền khung ảnh (RGB của #004F53)
    bg_color = (6, 88, 94)

    # Nếu ảnh grayscale → chuyển sang RGB
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)

    # Chuẩn hóa về 0–255
    arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
    arr = arr.astype(np.uint8)

    h, w, _ = arr.shape

    # Resize giữ nguyên tỷ lệ
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        arr, (new_w, new_h), interpolation=cv2.INTER_AREA
    )

    # Canvas nền đúng màu giao diện
    canvas = np.full(
        (target_size, target_size, 3),
        bg_color,
        dtype=np.uint8
    )

    # Canh giữa ảnh
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    # Hiển thị lên Tkinter
    img = Image.fromarray(canvas)
    img_tk = ImageTk.PhotoImage(img)
    panel.configure(image=img_tk)
    panel.image = img_tk




# ====================== LOAD ẢNH ======================
def select_image(reload=False):
    path = filedialog.askopenfilename(
        filetypes=[("DICOM files", "*.dcm;*.dicom")]
    )
    if not path:
        return

    if not reload:
        btn_start.place_forget()

    main_frame.pack(pady=30)

    result_title.place(relx=0.5, rely=0.72, anchor="center")
    result_class.place(relx=0.5, rely=0.77, anchor="center")
    result_conf.place(relx=0.5, rely=0.82, anchor="center")
    btn_reload.place(relx=0.5, rely=0.90, anchor="center")

    result_class.config(text="Đang xử lý...")
    result_conf.config(text="")
    root.update()

    # Ảnh gốc
    dcm = pydicom.dcmread(path)
    img_orig = dcm.pixel_array
    show_image(img_orig, labels["Ảnh gốc"])

    # Tiền xử lý
    temp_dir = "temp_processed"
    os.makedirs(temp_dir, exist_ok=True)
    proc_path = crop_mammogram_dicom(path, temp_dir, 224)
    img_proc = pydicom.dcmread(proc_path).pixel_array
    show_image(img_proc, labels["Ảnh tiền xử lý"])

    # Phân loại
    img_rgb = cv2.cvtColor(img_proc.astype(np.float32), cv2.COLOR_GRAY2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224))
    cls, conf = predict_image(img_rgb)

    result_class.config(text=cls)
    result_conf.config(text=f"ĐỘ TIN CẬY: {conf:.2f}%")


# ====================== NÚT ======================
btn_start = Button(
    root,
    text="Load image",
    command=lambda: select_image(False),
    bg="#004F53",
    fg="#FFD966",
    font=("Consolas", 20, "bold"),
    padx=28,
    pady=10,
    relief="flat",
)
btn_start.place(relx=0.5, rely=0.55, anchor="center")

btn_reload = Button(
    root,
    text="Load image",
    command=lambda: select_image(True),
    bg="#004F53",
    fg="#FFD966",
    font=("Consolas", 20, "bold"),
    padx=28,
    pady=10,
    relief="flat",
)
btn_reload.place_forget()


root.mainloop()
