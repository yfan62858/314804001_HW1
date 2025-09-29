## 1) Overview
- **(a) Image I/O** — Read BMP (8/24-bit, BI_RGB) and RAW (8-bit, arbitrary size) and convert to grayscale.  
  Also export the **center 10×10** values (CSV) and an enlarged **10×10 heatmap** (PGM/BMP).
- **(b) Gray-level transforms** via LUT:**Log**, **Gamma**, **Negative**
- **(c) Image scaling** with pixel-center alignment: **Nearest Neighbor** and **Bilinear**, including the five prescribed cases (i–v).
  
All functionality is contained in a single script: `ImageReading.py`.
