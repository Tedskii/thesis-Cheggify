# 🥚 Egg Size Classifier & Sorter (Raspberry Pi + YOLOv8 + Servo Control)

This project is a **smart egg classification and sorting system** built on a **Raspberry Pi 5**, using:

- **YOLOv8 segmentation model** for egg size detection (`S`, `M`, `L`, `XL`)
- **Flask Web Interface** for live control and monitoring
- **Arduino + Servo Motors** for mechanical egg sorting
- **PiCamera2** for image capture and live video stream

The system automates egg classification and physical separation into trays, designed for **Philippine poultry settings** where eggs are white.

---

## 🚀 Features

- **Live Video Feed** from PiCamera2
- **Automatic Egg Size Detection** via YOLOv8
- **Egg Tray Grid Mapping (5x6)** with size classification
- **Servo-Based Sorting** by size (`S=0`, `M=1`, `L=2`, `XL=3`)
- **Web Dashboard** (Flask):
  - Bootup status with Arduino connection progress
  - Capture live snapshot
  - View bounding box predictions
  - Select egg sizes for actuation
  - Reset moved or all servos
- **Shutdown Command** for Raspberry Pi from the dashboard

---

## 📂 Project Structure

