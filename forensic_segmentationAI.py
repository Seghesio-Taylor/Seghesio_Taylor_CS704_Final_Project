# Author: Taylor Seghesio
# Course: CS 704 Digital Forensics
# Institution: UNR
# Date: 15MAR2025


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import hashlib
import cv2
import psutil
from ultralytics import YOLO

# Detection threshold
CONF_THRESHOLD = 0.15
# Segmentation labeling settings for images
WEAPON_LABELS = {"firearm", "knife"}
BORDER_COLOR   = (0, 0, 255)
FONT_SCALE     = 0.5
FONT_THICKNESS = 2

def list_drives():
    drives = []
    for partition in psutil.disk_partitions(all=True):
        drives.append(partition.mountpoint)
    return drives

def train_model():
    print("Training initiated...")
    model = YOLO("yolo11m.pt")
    model.train(
        #training parameters:
        data="dataset.yaml",
        epochs=300,
        batch=8,
        patience=60,
        optimizer="AdamW",
        lr0=3e-4,
        lrf=1e-5,
        cos_lr=True,
        seed=42,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=5,
        pretrained=True,
        #augmentation parameters:
        imgsz=1280,
        mosaic=1.0,
        close_mosaic=10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mixup=0.1,
        copy_paste=0.1,
        erasing=0.6,
        #debug parameters:
        plots=True,
        verbose=True)
    best_weights = str(model.trainer.best)
    print(f"Best weights: {best_weights}")
    return best_weights


class DriveAnalysis:
    def __init__(self, volume_path, output_folder="output/hashed_images", model_weights="yolo11m.pt"):
        self.volume_path = volume_path
        self.image_files = []
        self.output_folder = output_folder
        self.model = YOLO(model_weights)
        os.makedirs(self.output_folder, exist_ok=True)

    def find_images(self):
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        for root, dirs, files in os.walk(self.volume_path):
            for file in files:
                if file.lower().endswith(image_extensions):
                    full_path = os.path.join(root, file)
                    self.image_files.append(full_path)
        print(f"[*] Found {len(self.image_files)} images on drive {self.volume_path}.")

    def detect_weapons(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None, False

        results = self.model(
            image_path,
            conf=CONF_THRESHOLD,
            iou=0.8,
            imgsz=1280,
            half=True,
            device=0,
            verbose=False)
        found = False

        for r in results:
            for box in r.boxes:
                label = self.model.names[int(box.cls)].lower()
                if label in WEAPON_LABELS:
                    found = True
                    self.label_image(img, box, label)

        return img, found

    def label_image(self,img, box, label):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), BORDER_COLOR, FONT_THICKNESS)
        cv2.putText(img, label.title(), (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, BORDER_COLOR, FONT_THICKNESS)

    def hash_and_save_image(self, image_path):
        with open(image_path, 'rb') as f:
            image_hash = hashlib.sha256(f.read()).hexdigest()
        processed_img, found = self.detect_weapons(image_path)
        if found and processed_img is not None:
            save_path = os.path.join(self.output_folder, f"{image_hash}.jpg")
            cv2.imwrite(save_path, processed_img)
            return save_path
        return None

    def process_images(self):
        detected_images = []
        for image_path in self.image_files:
            result = self.hash_and_save_image(image_path)
            if result is not None:
                detected_images.append(result)
        print(f"[*] Weapon detection complete. {len(detected_images)} images saved.")

    def run_analysis(self):
        self.find_images()
        self.process_images()

def main():
    print("Select mode:")
    print("1: Train yolo11m model using your dataset (Transfer Learning)")
    print("2: Run detection using optimized best weights")
    print("3: Run detection using pre-trained yolo11m model (no optimization)")
    mode = input("Enter your choice (1/2/3): ").strip()

    if mode == "1":
        model_weights = train_model()
        if input("Run detection now? (y/n): ").strip().lower() != "y":
            return
    elif mode == "2":
        model_weights = "runs/detect/train/weights/best.pt"
    elif mode == "3":
        model_weights = "yolo11m.pt"
    else:
        print("Invalid selection.")
        return

    drives = list_drives()
    if not drives:
        print("No drives found.")
        return
    print("Available drives:")
    for idx, drive in enumerate(drives, start=1):
        print(f"{idx}: {drive}")
    drive_choice = input("Enter the number for the drive you want to analyze: ").strip()
    try:
        selected_drive = drives[int(drive_choice) - 1]
    except (ValueError, IndexError):
        print("Invalid drive selection.")
        return

    print(f"Selected drive: {selected_drive}")
    analyzer = DriveAnalysis(selected_drive, model_weights=model_weights)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
