# -*- coding: utf-8 -*-
import os, json, random, warnings, math
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ======================
# 0) CÀI ĐẶT & MÔI TRƯỜNG
# ======================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)
cv2.setNumThreads(0)  # tránh OpenCV tạo quá nhiều luồng

# (Khuyến nghị) Cho phép TF cấp phát VRAM tăng dần (nếu có GPU NVIDIA)
try:
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception as e:
    print("[INFO] GPU memory growth setup skipped:", e)

# ======================
# 1) CẤU HÌNH
# ======================
data_dir   = r"D:\DO AN TOT NGHIEP - UTV\TAP-ANH-TRUONG-HOP-3\TH_3.3_Image_Dicom_224x224"
output_dir = r"D:\DO AN TOT NGHIEP - UTV\ResNet-50-tfdata\KET_QUA_TRUONG_HOP_3\KET-QUA-TRUONG-HOP-3.3\KET-QUA-TRAIN-100-EPOCHS-32-BATCHSIZE-(khong-class_weight)"
os.makedirs(output_dir, exist_ok=True)

img_size   = (224, 224)
batch_size = 32
epochs     = 100
seed       = 937
use_class_weight = False           # Bật/tắt cân bằng lớp
label_smoothing = 0.0         # 0.0–0.02 nếu lệch lớp mạnh

# Tuỳ chọn mixed precision nếu GPU hỗ trợ
use_mixed_precision = False
if use_mixed_precision:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

# tf.data options cho TRAIN: cho phép không tất định để tăng throughput I/O
train_options = tf.data.Options()
train_options.experimental_deterministic = False

# tf.data options cho EVAL (val/test): phải deterministic để tránh ngắt sớm khi cache
eval_options = tf.data.Options()
eval_options.experimental_deterministic = True

# Seed
import numpy.random as npr
random.seed(seed); np.random.seed(seed); npr.seed(seed); tf.random.set_seed(seed)

# ======================
# 2) DATAFRAME & SPLITS
# ======================
classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
filepaths, labels = [], []
for cls in classes:
    cls_dir = os.path.join(data_dir, cls)
    for f in os.listdir(cls_dir):
        if f.lower().endswith((".dcm", ".dicom")):
            filepaths.append(os.path.join(cls_dir, f))
            labels.append(cls)

df = pd.DataFrame({"filename": filepaths, "label": labels})
print("Tổng ảnh:", len(df), " | Số lớp:", len(classes), classes)

# Split: test 10%; val = 1/9 của phần còn lại -> 10% val, 80% train
train_val_df, test_df = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=seed)
train_df, val_df      = train_test_split(train_val_df, test_size=1/9, stratify=train_val_df["label"], random_state=seed)

class_names = classes
num_classes = len(class_names)
class2idx = {c:i for i,c in enumerate(class_names)}

# Chuyển label sang chỉ số
train_paths = train_df["filename"].values
val_paths   = val_df["filename"].values
test_paths  = test_df["filename"].values

train_labels = np.array([class2idx[l] for l in train_df["label"].values], dtype=np.int32)
val_labels   = np.array([class2idx[l] for l in val_df["label"].values],   dtype=np.int32)
test_labels  = np.array([class2idx[l] for l in test_df["label"].values],  dtype=np.int32)

# ===== LƯU class2idx song song .json & .xlsx =====
with open(os.path.join(output_dir, "class2idx.json"), "w", encoding="utf-8") as f:
    json.dump(class2idx, f, ensure_ascii=False, indent=2)
pd.DataFrame({"class_name": list(class2idx.keys()), "class_index": list(class2idx.values())}) \
  .to_excel(os.path.join(output_dir, "class2idx.xlsx"), index=False)

# ===== LƯU splits song song .json & .xlsx =====
splits_json = {
    "train": [{"filename": p, "label": int(l)} for p, l in zip(train_paths, train_labels)],
    "val":   [{"filename": p, "label": int(l)} for p, l in zip(val_paths,   val_labels)],
    "test":  [{"filename": p, "label": int(l)} for p, l in zip(test_paths,  test_labels)],
}
with open(os.path.join(output_dir, "splits.json"), "w", encoding="utf-8") as f:
    json.dump(splits_json, f, ensure_ascii=False, indent=2)

with pd.ExcelWriter(os.path.join(output_dir, "splits.xlsx")) as writer:
    pd.DataFrame(splits_json["train"]).to_excel(writer, sheet_name="train", index=False)
    pd.DataFrame(splits_json["val"]).to_excel(writer,   sheet_name="val",   index=False)
    pd.DataFrame(splits_json["test"]).to_excel(writer,  sheet_name="test",  index=False)

# ======================
# 3) TIỀN XỬ LÝ DICOM (KHÔNG can thiệp hình học đã có)
# ======================
def dicom_to_uint8_rgb_no_geom(dcm, target_size=(224,224)):
    # 0) Đọc raw (Stored Values)
    raw = dcm.pixel_array.astype(np.float32)

    # 1) Tạo mask padding theo DICOM (trên raw)
    pad_mask = None
    pv  = getattr(dcm, "PixelPaddingValue", None)
    prl = getattr(dcm, "PixelPaddingRangeLimit", None)
    if pv is not None:
        pv = float(pv)
        if prl is not None:
            prl = float(prl)
            lo, hi = (pv, prl) if pv <= prl else (prl, pv)
            pad_mask = (raw >= lo) & (raw <= hi)
        else:
            pad_mask = (raw == pv)

    # 2) VOI LUT
    img = apply_voi_lut(raw, dcm).astype(np.float32)

    # 3) Đảo MONOCHROME1 sau LUT
    if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img

    # 4) Ép vùng padding (đã xác định trên raw) về min sau LUT
    if pad_mask is not None:
        img[pad_mask] = img.min()

    # 5) Chuẩn hoá 0..255 (uint8)
    imin, imax = float(img.min()), float(img.max())
    if imax > imin:
        img = (img - imin) / (imax - imin)
    else:
        img = np.zeros_like(img, dtype=np.float32)
    img = (img * 255.0).astype(np.uint8)

    # 6) Letterbox về target_size (nền đen, không méo)
    h, w = img.shape
    th, tw = target_size
    if (h, w) != (th, tw):
        scale = min(th / h, tw / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((th, tw), dtype=np.uint8)
        y0 = (th - nh) // 2
        x0 = (tw - nw) // 2
        canvas[y0:y0+nh, x0:x0+nw] = resized
        img = canvas

    # 7) Gray -> RGB float32 (0..255) để hợp với preprocess_input(ResNet50)
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.float32)
    return rgb

def load_dicom_as_rgb(path, target_size=(224,224)):
    # Dùng tf.py_function → nhận bytes, cần decode thủ công
    p = path.numpy().decode("utf-8", errors="ignore")
    try:
        dcm = pydicom.dcmread(p, force=True)
        img = dicom_to_uint8_rgb_no_geom(dcm, target_size)
    except Exception as e:
        # Không để lỗi làm dừng training: trả ảnh đen, đồng thời log cảnh báo
        tf.print("[WARN] skip bad dicom:", p, e)
        img = np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)
    return img

def preprocess_fn(path, label):
    img = tf.py_function(func=load_dicom_as_rgb, inp=[path], Tout=tf.float32)
    img.set_shape((img_size[0], img_size[1], 3))
    img = preprocess_input(img)  # ResNet50: 'caffe' mode (RGB->BGR + zero-center ImageNet)
    label = tf.one_hot(tf.cast(label, tf.int32), depth=num_classes, dtype=tf.float32)
    return img, label

# ======================
# 4) DATASET BUILDERS
# ======================
# Chỉ cache val/test ra FILE; train KHÔNG cache để tránh tốn đĩa
train_cache_path = None
val_cache_path   = os.path.join(output_dir, "val.cache")
test_cache_path  = os.path.join(output_dir, "test.cache")

# Giảm song song hoá để tránh bùng RAM; có thể điều chỉnh 2–8 tuỳ máy
num_map_calls = 4

def build_ds(paths, labels, training=False, cache_path=None, num_calls=4):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=min(len(paths), 10000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(preprocess_fn, num_parallel_calls=num_calls)
    if cache_path:
        ds = ds.cache(cache_path)  # KHÔNG dùng cho train trong cấu hình này
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.with_options(train_options if training else eval_options)
    return ds

def safe_remove(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
            print(f"[INFO] Removed old cache: {path}")
    except Exception as e:
        print("[WARN] Could not remove old cache:", path, e)

def build_ds_eval(paths, labels, cache_path=None, num_calls=4):
    """
    Eval dataset CHUẨN: take -> cache -> repeat + PRIME CACHE
    - deterministic=True cho eval
    - tránh cảnh báo 'did not fully read the dataset being cached'
    """
    n = len(paths)
    assert n > 0, "Eval set rỗng!"
    steps = math.ceil(n / batch_size)

    base = tf.data.Dataset.from_tensor_slices((paths, labels))
    base = base.map(preprocess_fn, num_parallel_calls=num_calls)
    base = base.batch(batch_size, drop_remainder=False)

    # Xoá cache cũ (nếu có)
    if cache_path:
        safe_remove(cache_path)

    # PRIME CACHE: đổ đầy cache trước khi dùng trong fit/predict
    if cache_path:
        prime = base.take(steps).cache(cache_path)
        prime = prime.prefetch(tf.data.AUTOTUNE).with_options(eval_options)
        for _ in prime:
            pass  # chạy đúng 'steps' batch để lấp đầy cache

    # Dataset eval chính thức
    ds = base.take(steps)
    if cache_path:
        ds = ds.cache(cache_path)
    ds = ds.repeat()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.with_options(eval_options)
    return ds, steps

# Build datasets
train_ds = build_ds(train_paths, train_labels, training=True,
                    cache_path=train_cache_path, num_calls=num_map_calls)

val_ds,  val_steps  = build_ds_eval(val_paths,  val_labels,
                                    cache_path=val_cache_path,  num_calls=num_map_calls)
test_ds, test_steps = build_ds_eval(test_paths, test_labels,
                                    cache_path=test_cache_path, num_calls=num_map_calls)

print(f"[VAL] samples={len(val_paths)}, batch_size={batch_size}, steps={val_steps}")
print(f"[TEST] samples={len(test_paths)}, batch_size={batch_size}, steps={test_steps}")

# ======================
# 5) MÔ HÌNH RESNET50 (đóng băng backbone)
# ======================
base = ResNet50(weights="imagenet", include_top=False, input_shape=img_size + (3,))
base.trainable = False

x = layers.GlobalAveragePooling2D()(base.output)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
logits = layers.Dense(num_classes)(x)
out = layers.Activation("softmax", dtype="float32")(logits)  # cưỡng float32 nếu mixed precision

model = models.Model(inputs=base.input, outputs=out)
model.compile(
    optimizer=optimizers.Adam(1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
    metrics=["accuracy"]
)
model.summary()

# ======================
# 6) CALLBACKS
# ======================
ckpt_path = os.path.join(output_dir, "ResNet50_model.keras")
ckpt = callbacks.ModelCheckpoint(
    ckpt_path, monitor="val_loss", mode="min", save_best_only=True
)
early = callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10, restore_best_weights=True)
reduce = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1)

# ======================
# 7) CLASS WEIGHTS (tuỳ chọn)
# ======================
class_weight = None
if use_class_weight:
    counts = Counter(train_labels.tolist())
    total  = sum(counts.values())
    class_weight = {i: total/(num_classes*counts.get(i, 1)) for i in range(num_classes)}
    print("Class weights:", class_weight)

# ======================
# 8) TRAINING
# ======================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    validation_steps=val_steps,   # <-- QUAN TRỌNG khi val_ds .repeat()
    epochs=epochs,
    callbacks=[ckpt, early, reduce],
    class_weight=class_weight,
    verbose=1
)

# ======================
# 9) VẼ CURVE
# ======================
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir,"ResNet50_training_curves.png"))
plt.close()

# ===== LƯU HISTORY song song JSON & XLSX =====
hist_df = pd.DataFrame(history.history)
hist_df.to_excel(os.path.join(output_dir,"training_history.xlsx"), index=False)
with open(os.path.join(output_dir,"training_history.json"), "w", encoding="utf-8") as f:
    json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, ensure_ascii=False, indent=2)

# ======================
# 10) ĐÁNH GIÁ TEST
# ======================
print("\nĐánh giá trên test set…")
best_model = tf.keras.models.load_model(ckpt_path)

# test_ds là dataset .repeat() -> PHẢI truyền steps để đọc đúng 1 vòng
preds = best_model.predict(test_ds, steps=test_steps, verbose=1)   # shape [N_test, num_classes]
y_pred = np.argmax(preds, axis=1)
y_true = test_labels

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"confusion_matrix.png"))
plt.close()

# Classification report
report_dict = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
report_txt  = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report_txt)

# ===== LƯU REPORT song song JSON/TXT/XLSX =====
with open(os.path.join(output_dir,"classification_report.json"), "w", encoding="utf-8") as f:
    # convert numpy types to native
    def _num(v):
        try:
            return float(v)
        except Exception:
            return v
    rep = {k: {kk: _num(vv) for kk, vv in v.items()} if isinstance(v, dict) else v
           for k, v in report_dict.items()}
    json.dump(rep, f, ensure_ascii=False, indent=2)

with open(os.path.join(output_dir,"classification_report.txt"),"w",encoding="utf-8") as f:
    f.write(report_txt)

pd.DataFrame(report_dict).transpose().to_excel(
    os.path.join(output_dir,"classification_report.xlsx")
)

# ===== LƯU DỰ ĐOÁN CHI TIẾT song song JSON & XLSX =====
pred_rows = []
for p, t, probs in zip(test_paths, y_true, preds):
    row = {"filename": p, "true_label_idx": int(t), "true_label_name": class_names[int(t)],
           "pred_label_idx": int(np.argmax(probs)), "pred_label_name": class_names[int(np.argmax(probs))]}
    # thêm xác suất từng lớp
    for i, cname in enumerate(class_names):
        row[f"prob_{i}_{cname}"] = float(probs[i])
    pred_rows.append(row)

pred_df = pd.DataFrame(pred_rows)
pred_df.to_excel(os.path.join(output_dir, "predictions.xlsx"), index=False)

with open(os.path.join(output_dir, "predictions.json"), "w", encoding="utf-8") as f:
    json.dump(pred_rows, f, ensure_ascii=False, indent=2)

print(f"\n✅ Tất cả kết quả đã lưu trong: {output_dir}")
