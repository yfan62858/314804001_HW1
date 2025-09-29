## 1) Overview
-  Image I/O** — Read BMP (8/24-bit, BI_RGB) and RAW (8-bit, arbitrary size) and convert to grayscale.  
  Also export the **center 10×10** values (CSV) and an enlarged **10×10 heatmap** (PGM/BMP).
- Gray-level transforms** via LUT:**Log**, **Gamma**, **Negative**
- Image scaling** with pixel-center alignment: **Nearest Neighbor** and **Bilinear**, including the five prescribed cases.
  
All functionality is contained in a single script: `ImageReading.py`.

## 2) Usage

### (a) Read images & export center 10×10
```bash
python ImageReading.py read
```

### (b) Gray-level transforms (Negative / Log / Gamma)
```bash
python ImageReading.py xform
```

### (c) Scaling (Nearest / Bilinear)
```bash
python ImageReading.py resize
```

## 3) GUI

GUI is contained in a single script: `mini_photoshop.py`.

```bash
python mini_photoshop/main.py
```
### Function
- Open file (bmp / raw)
- Save as BMP
- Image enhancement toolkit (log / gamma / negative / Curve)
- Image sampling (nearest / bilinear)
- Reset
