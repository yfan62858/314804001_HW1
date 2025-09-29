#!/usr/bin/env python3                    
# -*- coding: utf-8 -*-                   

"""(a) Image reading : display them / 10x10 pixel values
(b) Image enhancement toolkit : log-transformg / gamma-transform, / image negative
(c) Image downsampling and upsampling : bilinear and nearest-neighbor interpolation methods"""

import os                                  
import sys                                 
import struct                              
import math                                
import argparse                            
from typing import List, Optional


# ------------------------------ 共用：灰階影像容器 ------------------------------
class Image:
    """保存灰階影像的寬高與像素"""
    def __init__(self, w: int, h: int, pix: bytearray):
        self.w = w                          
        self.h = h                          
        self.pix = pix                      


# ============================== (a) 讀 RAW ==============================
def _parse_wh_from_filename(path: str) -> Optional[tuple[int, int]]:
    """抓 WxH，lena_640x480.raw → (640,480)。抓不到回傳 None"""
    import re                               
    name = os.path.basename(path)           
    m = re.search(r'(\d+)\s*[xX]\s*(\d+)', name)  
    if m:                                   
        w = int(m.group(1))                 
        h = int(m.group(2))                 
        if w > 0 and h > 0:                 
            return (w, h)                  
    return None                            


def load_raw(path: str, w: Optional[int] = None, h: Optional[int] = None) -> Image:
    """讀 8-bit RAW（單通道）。優先順序：CLI 指定 → 檔名 → 正方形 → 常見解析度表；都不行就拋錯。"""
    with open(path, "rb") as f:             # 以二進位模式打開
        data = f.read()                     
    n = len(data)                           

    if w and h:                             # 若使用者有明確提供寬高
        if w * h != n:                      
            raise ValueError(f"RAW size mismatch: got {n} bytes, w*h={w*h}")
        return Image(w, h, bytearray(data)) 

    wh = _parse_wh_from_filename(path)      # 嘗試從檔名抓寬高
    if wh:                                  
        w2, h2 = wh                         
        if w2 * h2 != n:                    
            raise ValueError(f"RAW size mismatch: got {n} bytes, w*h={w2*h2} from filename")
        return Image(w2, h2, bytearray(data))  

    r = int(math.isqrt(n))                  # 嘗試平方根（完全平方數 → 視為正方形）
    if r * r == n:                          
        return Image(r, r, bytearray(data)) 

    common = [                              # 常見寬高表（可視需要擴充）
        (640, 480), (800, 600), (1024, 768),
        (1280, 720), (1280, 960), (1920, 1080),
    ]
    for (cw, ch) in common:                 # 逐一嘗試常見解析度
        if cw * ch == n:                    
            return Image(cw, ch, bytearray(data)) 

    # 全部方法都失敗 → 需要使用者指定
    raise ValueError(
        f"Cannot infer RAW size for {path}. Please pass --raww --rawh or rename like *_640x480.raw"
    )


# ============================== (a) 讀 BMP 然後轉灰階 ==============================
# 小工具：讀取 BMP 標頭（little-endian）
def _rd_u16(b, off): return struct.unpack_from("<H", b, off)[0]   # 讀 2-byte 無號整數
def _rd_i32(b, off): return struct.unpack_from("<i", b, off)[0]   # 讀 4-byte 有號整數
def _rd_u32(b, off): return struct.unpack_from("<I", b, off)[0]   # 讀 4-byte 無號整數

def load_bmp_to_gray(path: str) -> Image:
    """只支援 BI_RGB（無壓縮）的 8-bit palette 或 24-bit BGR；轉成灰階 Image。"""
    with open(path, "rb") as f:             
        B = f.read()                        

    if _rd_u16(B, 0) != 0x4D42:             # 檔頭 2 bytes 必須是 'BM'(0x4D42)
        raise ValueError("Not a BMP")       

    offBits = _rd_u32(B, 10)                # 像素資料在檔案中的偏移
    dibSize = _rd_u32(B, 14)                # DIB 區塊大小（至少 40）
    if dibSize < 40:                        
        raise ValueError("DIB < 40 unsupported")  #格式太舊，不支援
    W  = _rd_i32(B, 18)                     
    Hs = _rd_i32(B, 22)                     
    planes = _rd_u16(B, 26); _ = planes     
    bpp  = _rd_u16(B, 28)                   # 每像素位元數（支援 8 或 24）
    comp = _rd_u32(B, 30)                   # 壓縮方式（0=BI_RGB）
    _ = _rd_u32(B, 34); _ = _rd_i32(B, 38); _ = _rd_i32(B, 42)  
    clrUsed = _rd_u32(B, 46)                

    if comp != 0:                           # 僅支援無壓縮
        raise ValueError("Only BI_RGB (uncompressed)")
    H = abs(Hs)                            
    bottom_up = Hs > 0          
    if W <= 0 or H <= 0:                   
        raise ValueError("Invalid BMP size")

    pal = None                             
    if bpp == 8:                           
        pal = []                            
        pal_off = 14 + dibSize              
        n = clrUsed if clrUsed else 256     
        for i in range(n):                  
            b, g, r, a = struct.unpack_from("<BBBB", B, pal_off + 4*i)
            pal.append((r, g, b))           # 存成 (R,G,B) 方便轉灰階

    def toY(r, g, b):
        # 整數近似：Y ≈ 0.299R + 0.587G + 0.114B → (77R + 150G + 29B + 128) >> 8
        y = (77*r + 150*g + 29*b + 128) >> 8
        return 0 if y < 0 else (255 if y > 255 else y)  # 夾在 0..255

    def stride(bits):                       # 每列位元組數需 4-byte 對齊（BMP 規定）
        return ((W * bits + 31) // 32) * 4  
    
    out = bytearray(W * H)                  # 建立灰階輸出緩衝
    p = offBits                             # 像素資料目前位移
    if bpp == 8:                            # 8-bit palette 模式
        s = stride(8)                       # 每列含 padding 的長度
        for y in range(H):                  
            row = B[p:p+s]; p += s          # 切出該列的原始資料（含 padding）
            dy = (H - 1 - y) if bottom_up else y    # bottom-up 需倒著寫入
            for x in range(W):              
                r, g, b = pal[row[x]]       # 查 palette 得到 RGB
                out[dy*W + x] = toY(r, g, b)  # 轉灰階
    elif bpp == 24:                         # 24-bit BGR 模式
        s = stride(24)                      # 每列含 padding 的長
        for y in range(H):
            row = B[p:p+s]; p += s
            dy = (H - 1 - y) if bottom_up else y
            for x in range(W):
                b = row[3*x + 0]; g = row[3*x + 1]; r = row[3*x + 2]  # 取 B,G,R
                out[dy*W + x] = toY(r, g, b)  # 轉灰階
    else:
        raise ValueError("BMP bpp not 8/24") # 其他 bpp（如 32）本工具不支援
    return Image(W, H, out)                  # 回傳灰階影像


# -------------------- 輸出：PGM 與 24-bit BMP --------------------
def save_pgm(img: Image, path: str):
    """以 PGM (P5, binary) 格式儲存：簡潔、相容性好、適合驗證矩陣內容。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)   # 確保資料夾存在
    with open(path, "wb") as f:                         # 以二進位寫入
        f.write(f"P5\n{img.w} {img.h}\n255\n".encode("ascii"))  # Netpbm 標頭
        f.write(img.pix)                                 # 直接寫 raw 灰階位元組

def _wr_u16(f, v): f.write(struct.pack("<H", v))        # 小工具：寫 2-byte little-endian
def _wr_u32(f, v): f.write(struct.pack("<I", v))        # 小工具：寫 4-byte little-endian

def save_bmp24_gray(img: Image, path: str):
    """輸出 24-bit BMP：把灰階值複製到 B/G/R 三通道，方便預設看圖工具顯示。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)   
    W, H = img.w, img.h                                 
    stride = ((W*3 + 3) // 4) * 4                       
    total = stride * H                                  
    with open(path, "wb") as f:                         
        # BITMAPFILEHEADER（14 bytes）
        f.write(b'BM')                                   
        _wr_u32(f, 14 + 40 + total)                      # 檔案總大小
        _wr_u16(f, 0); _wr_u16(f, 0)                     # 保留欄位
        _wr_u32(f, 14 + 40)                              # 像素資料起始偏移
        # BITMAPINFOHEADER（40 bytes）
        _wr_u32(f, 40)                                   # 資訊頭大小
        _wr_u32(f, W); _wr_u32(f, H)                     # 注意：用正高 → bottom-up
        _wr_u16(f, 1); _wr_u16(f, 24)                    # 平面=1；24bpp
        _wr_u32(f, 0)                                    # BI_RGB（無壓縮）
        _wr_u32(f, total)                                # 影像資料大小
        _wr_u32(f, 2835); _wr_u32(f, 2835)               # 解析度（72DPI≈2835 px/m）
        _wr_u32(f, 0); _wr_u32(f, 0)                     # 調色盤資訊（不用）
        # 逐列（bottom-up）輸出像素，並自動補 padding
        row = bytearray(stride)                           # 暫存一列（含 padding 的零）
        for y in range(H-1, -1, -1):                      # 從最後一列寫到第一列
            for x in range(W):
                v = img.pix[y*W + x]                      # 取灰階值
                i = x*3                                   # 此像素在列中的 B 起始索引
                row[i] = v; row[i+1] = v; row[i+2] = v    # B, G, R 都填 v
            f.write(row)                                  


# -------------------- (a) 中央 10×10：CSV 與放大視覺化 --------------------
def save_center10_csv(img: Image, path: str):
    """把中心 10×10 的像素值存成 CSV。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)    
    n = 10                                             
    r0 = (img.h - n) // 2                              # 中心起始列
    c0 = (img.w - n) // 2                              # 中心起始行
    with open(path, "w", encoding="utf-8") as f:       # 文字寫入 CSV
        for r in range(n):                             
            row = [str(img.pix[(r0+r)*img.w + (c0+c)]) for c in range(n)]  # 取 10 個
            f.write(",".join(row) + "\n")              

def make_center10_heat(img: Image, scale: int = 32) -> Image:
    """把中心 10×10 放大（最近鄰）到 (10*scale)^2。"""
    n = 10                                             
    r0 = (img.h - n) // 2                              
    c0 = (img.w - n) // 2                              
    W, H = n*scale, n*scale                            
    out = bytearray(W * H)                             # 建立輸出緩衝
    for y in range(H):                                 
        sy = y // scale                                
        for x in range(W):                             
            sx = x // scale   
            out[y*W + x] = img.pix[(r0+sy)*img.w + (c0+sx)]  # 複製對應灰階
    # 畫黑色格線（置 0）
    for k in range(0, H, scale):                       # 每 scale 畫一條水平線
        for x in range(W): out[k*W + x] = 0
    for k in range(0, W, scale):                       
        for y in range(H): out[y*W + k] = 0
    return Image(W, H, out)                            


# ============================== (b) 灰階強度轉換 ==============================
def clamp8(x: float) -> int:
    """把值夾在 0..255 並四捨五入，避免運算誤差造成越界。"""
    x = 0 if x < 0 else (255 if x > 255 else x)        
    return int(x + 0.5)                                

def lut_negative() -> bytes:
    """負片：s = 255 - r。回傳 256 長度的 LUT（bytes）。"""
    return bytes([255 - i for i in range(256)])

def lut_log(c: float = 45.0) -> bytes:
    """對數轉換：s = c * log(1 + r)。用 log1p 改善小值精度。"""
    return bytes([clamp8(c * math.log1p(i)) for i in range(256)])

def lut_gamma(g: float = 0.6) -> bytes:
    """Gamma：s = 255 * (r/255)^γ。γ<1 變亮；γ>1 變暗。"""
    return bytes([clamp8(255.0 * ((i/255.0) ** g)) for i in range(256)])

def apply_lut(img: Image, lut: bytes) -> Image:
    """查表套用：對每個像素執行 dst[i] = lut[src[i]]，比逐像素算數學函式快非常多。"""
    W, H = img.w, img.h                               
    out = bytearray(W * H)                            
    p = img.pix                                       
    for i in range(W * H):                            
        out[i] = lut[p[i]]                           
    return Image(W, H, out)                         


# ============================== (c) 影像縮放：像素中心對齊 ==============================
def _map_center(x: int, sx: float) -> float:
    """把目標像素中心 (x+0.5) 映射回原圖座標：fx = (x+0.5)*sx - 0.5（避免半像素系統偏移）。"""
    return (x + 0.5) * sx - 0.5

def resize_nearest(img: Image, W: int, H: int) -> Image:
    """最近鄰縮放：把映射座標四捨五入到最近像素。"""
    sw, sh = img.w, img.h                              
    out = bytearray(W * H)                             
    sx = sw / W                                       
    sy = sh / H                                        
    for y in range(H):                                 
        fy = _map_center(y, sy)                        # 對應原圖 y 的實數座標
        syc = int(round(fy))                           # 最近鄰列索引
        syc = 0 if syc < 0 else (sh-1 if syc >= sh else syc)  
        for x in range(W):                            
            fx = _map_center(x, sx)                   
            sxc = int(round(fx))                     
            sxc = 0 if sxc < 0 else (sw-1 if sxc >= sw else sxc)  
            out[y*W + x] = img.pix[syc*sw + sxc]       
    return Image(W, H, out)                           

def resize_bilinear(img: Image, W: int, H: int) -> Image:
    """雙線性縮放：水平與垂直各做一次線性內插，總共四個鄰點加權平均。"""
    sw, sh = img.w, img.h                            
    out = bytearray(W * H)                           
    sx = sw / W                                       
    sy = sh / H                                      
    for y in range(H):                              
        fy = (y + 0.5) * sy - 0.5                      
        y0 = int(math.floor(fy))                       # 下界列
        y1 = min(sh - 1, y0 + 1)                       # 上界列（不可超界）
        y0 = max(0, y0)                                # 夾住下界
        wy = fy - y0                                   # 垂直權重（0..1）
        for x in range(W):              
            fx = (x + 0.5) * sx - 0.5                  # 原圖 x 實數座標
            x0 = int(math.floor(fx))                   
            x1 = min(sw - 1, x0 + 1)                  
            x0 = max(0, x0)                          
            wx = fx - x0                              
            p00 = img.pix[y0*sw + x0]                  # 左上
            p10 = img.pix[y0*sw + x1]                  # 右上
            p01 = img.pix[y1*sw + x0]                  # 左下
            p11 = img.pix[y1*sw + x1]                  # 右下
            top = p00*(1-wx) + p10*wx                  # 先水平內插「上」一列
            bot = p01*(1-wx) + p11*wx                  # 再水平內插「下」一列
            v = int(round(top*(1-wy) + bot*wy))        # 垂直內插並四捨五入
            out[y*W + x] = 0 if v < 0 else (255 if v > 255 else v)  
    return Image(W, H, out)                             

def _resize(img: Image, w: int, h: int, method: str) -> Image:
    """小幫手：根據 method 呼叫對應的縮放實作。"""
    return resize_nearest(img, w, h) if method == "nearest" else resize_bilinear(img, w, h)

def _save(img: Image, out_dir: str, tag: str):
    """同時輸出 BMP 與 PGM，任何環境都能打開。"""
    save_bmp24_gray(img, os.path.join(out_dir, f"{tag}.bmp"))  # 存 BMP（24-bit 灰階）
    save_pgm(img,         os.path.join(out_dir, f"{tag}.pgm"))  # 存 PGM（純灰階）


# ============================== 共用：自動載入與命名 ==============================
def load_auto(path: str, w: Optional[int] = None, h: Optional[int] = None) -> Image:
    """依副檔名挑選讀取器：.raw → load_raw（可帶 w/h）、.bmp → 解析 BMP。"""
    ext = os.path.splitext(path)[1].lower()            # 取副檔名小寫
    if ext == ".raw": return load_raw(path, w, h)      # RAW：傳遞 w/h（可能是 None）
    if ext == ".bmp": return load_bmp_to_gray(path)    # BMP：轉灰階
    raise ValueError(f"Unsupported extension: {ext} (use .bmp or .raw)")  # 其他格式不支援

TEST_FILES: List[str] = [                              # 預設要批次處理的六張檔名
    "baboon.bmp", "boat.bmp", "F16.bmp", "lena.raw", "peppers.raw", "goldhill.raw"
]

def nice_name(input_path: str) -> str:
    """RAW 檔在輸出資料夾名稱加 _raw 後綴，避免與同名 BMP 混淆。"""
    base = os.path.splitext(os.path.basename(input_path))[0]    # 取純檔名
    return base + ("_raw" if input_path.lower().endswith(".raw") else "")  # RAW 就加尾綴


# ============================== (a) / (b) / (c) 流程函式 ==============================
RAW_W: Optional[int] = None                   # 供 CLI 指定的 RAW 寬（若未指定為 None）
RAW_H: Optional[int] = None                   # 供 CLI 指定的 RAW 高（若未指定為 None）

def do_read(in_path: str, out_root: str):
    """(a) 讀檔→灰階；輸出整圖 + 中央 10×10 的 CSV 與放大熱度圖。"""
    sub = f"a_{nice_name(in_path)}"                        # 子資料夾名稱（含 a_ 前綴）
    out_dir = os.path.join(out_root, sub)                  # 最終輸出資料夾
    os.makedirs(out_dir, exist_ok=True)                    
    img = load_auto(in_path, RAW_W, RAW_H)                
    save_pgm(img,               os.path.join(out_dir, "loaded_grayscale.pgm"))            
    save_bmp24_gray(img,        os.path.join(out_dir, "loaded_grayscale.bmp"))          
    save_center10_csv(img,      os.path.join(out_dir, "center10x10.csv"))              
    heat = make_center10_heat(img, 32)                    
    save_pgm(heat,              os.path.join(out_dir, "center10x10_heatmap.pgm"))        
    save_bmp24_gray(heat,       os.path.join(out_dir, "center10x10_heatmap.bmp"))        
    print(f"[A] {in_path} -> {out_dir}")                   

def do_xform(in_path: str, out_root: str, typ: str, cval: float, gamma: float):
    """(b) 單張：使用 LUT 套用 negative / log(c) / gamma(γ)。"""
    sub = f"b_{nice_name(in_path)}"                        
    out_dir = os.path.join(out_root, sub)                 
    os.makedirs(out_dir, exist_ok=True)                    
    img = load_auto(in_path, RAW_W, RAW_H)               
    if   typ == "negative": out, tag = apply_lut(img, lut_negative()), "neg"             
    elif typ == "log":      out, tag = apply_lut(img, lut_log(cval)), f"log_c{cval:g}"   
    elif typ == "gamma":    out, tag = apply_lut(img, lut_gamma(gamma)), f"gamma_{gamma:g}"  
    else: raise ValueError(f"unknown type: {typ}")         
    _save(out, out_dir, tag)                              
    print(f"[B] {typ} {in_path} -> {out_dir}")            

def do_resize(in_path: str, out_root: str, w: int, h: int, method: str, suite: bool = False):
    """(c) 單張：一般縮放或輸出題目指定的五種情境（--suite）。"""
    sub = f"c_{nice_name(in_path)}"                      
    out_dir = os.path.join(out_root, sub)                
    os.makedirs(out_dir, exist_ok=True)                  
    src = load_auto(in_path, RAW_W, RAW_H)               

    if not suite:                                         
        out = _resize(src, w, h, method)                  
        tag = f"resize_{w}x{h}_{method}"                   
        _save(out, out_dir, tag)                           
        print(f"[C] {method} {w}x{h} {in_path} -> {out_dir}")  
        return                                         

    for m in ("nearest", "bilinear"):                      # 整套五情境：每種方法各做一次
        img_128 = _resize(src, 128, 128, m);   _save(img_128, out_dir, f"i_512to128_{m}")        # i) 512→128
        img_32  = _resize(src, 32, 32, m);     _save(img_32,  out_dir, f"ii_512to32_{m}")        # ii) 512→32
        img_32_to_512 = _resize(img_32, 512, 512, m); _save(img_32_to_512, out_dir, f"iii_32to512_{m}")  # iii) 32→512
        img_1024x512 = _resize(src, 1024, 512, m);    _save(img_1024x512, out_dir, f"iv_512to1024x512_{m}")  # iv) 512→1024×512
        img_256x512 = _resize(img_128, 256, 512, m);  _save(img_256x512, out_dir, f"v_128to256x512_{m}")     # v) 128→256×512
    print(f"[C] suite x2 methods {in_path} -> {out_dir}")  # 印出完成


# ============================== 一次跑全部 ==============================
def run_all_a(data_dir="data", out_root="output_a"):
    """跑 (a) 六張檔：data/ 下的預設檔名，逐一輸出到 output_a。"""
    for f in TEST_FILES:                                  
        do_read(os.path.join(data_dir, f), out_root)       

def run_all_b(data_dir="data", out_root="output_b"):
    """跑 (b) 六張檔：每張做 negative / log / gamma(0.6,1.5)。"""
    for f in TEST_FILES:
        p = os.path.join(data_dir, f)
        do_xform(p, out_root, "negative", 45.0, 0.5)      
        do_xform(p, out_root, "log",      45.0, 0.5)       
        for g in (0.6, 1.5):                               
            do_xform(p, out_root, "gamma", 45.0, g)

def run_all_c(data_dir="data", out_root="output_c"):
    """跑 (c) 六張檔：對每張輸出五情境（兩種方法都會存）。"""
    for f in TEST_FILES:
        p = os.path.join(data_dir, f)
        do_resize(p, out_root, 1024, 1024, "nearest", suite=True)  


# ============================== CLI ==============================
def main():
    """命令列介面：read / xform / resize / runall 四種子命令。"""
    ap = argparse.ArgumentParser(description="Assignment-1 (a)(b)(c) tool")  
    sub = ap.add_subparsers(dest="cmd", required=True)                       

    # (a) read
    sp = sub.add_parser("read", help="(a) read one (BMP/RAW) -> grayscale + center 10x10")
    sp.add_argument("input", nargs="?", help="path to image (default: run all six from ./data)")  # 可不給→跑全部
    sp.add_argument("--outroot", default="output_a")                                             
    sp.add_argument("--raww", type=int, help="width for RAW (optional)")                          
    sp.add_argument("--rawh", type=int, help="height for RAW (optional)")                         

    # (b) xform
    sp = sub.add_parser("xform", help="(b) gray-level transform: negative/log/gamma")
    sp.add_argument("input", nargs="?", help="path to image (default: run all six from ./data)")
    sp.add_argument("--outroot", default="output_b")
    sp.add_argument("--type", choices=["negative","log","gamma"], help="required if input is given")
    sp.add_argument("--c", type=float, default=45.0)                                            
    sp.add_argument("--gamma", type=float, default=0.5)                                           
    sp.add_argument("--raww", type=int, help="width for RAW (optional)")
    sp.add_argument("--rawh", type=int, help="height for RAW (optional)")

    # (c) resize
    sp = sub.add_parser("resize", help="(c) resize (nearest/bilinear) or --suite for 5 required cases")
    sp.add_argument("input", nargs="?", help="path to image (default: run all six from ./data)")
    sp.add_argument("--outroot", default="output_c")
    sp.add_argument("--w", type=int, default=1024)                                             
    sp.add_argument("--h", type=int, default=1024)                                                
    sp.add_argument("--method", choices=["nearest","bilinear"], default="nearest")               
    sp.add_argument("--suite", action="store_true", help="run the 5 cases (i–v) with both methods")
    sp.add_argument("--raww", type=int, help="width for RAW (optional)")
    sp.add_argument("--rawh", type=int, help="height for RAW (optional)")

    # one-shot：全部 a+b+c 一次跑
    sub.add_parser("runall", help="run a+b+c for all six test images (data/ -> output_*)")

    args = ap.parse_args()                                

    # 把 RAW 尺寸（若有）設成全域，供 load_auto 用（避免每個函式都要加參數）
    global RAW_W, RAW_H
    RAW_W = getattr(args, "raww", None)                     
    RAW_H = getattr(args, "rawh", None)                   

    if args.cmd == "read":                                  # 子命令：read
        if args.input:                                      # 若有指定單一檔案
            do_read(args.input, args.outroot)               # 跑 (a) 單張
        else:
            run_all_a(out_root=args.outroot)                # 否則跑六張預設檔

    elif args.cmd == "xform":                             
        if args.input:                                      
            if not args.type: ap.error("xform: --type is required when a single input is given")  
            do_xform(args.input, args.outroot, args.type, args.c, args.gamma)                     
        else:
            run_all_b(out_root=args.outroot)               

    elif args.cmd == "resize":                             
        if args.input:                                     
            do_resize(args.input, args.outroot, args.w, args.h, args.method, suite=args.suite)  
        else:
            run_all_c(out_root=args.outroot)                

    else:                                                   # 子命令：runall
        run_all_a()                                         
        run_all_b()                                         
        run_all_c()                                         

if __name__ == "__main__":                                  # 直接執行檔案時才進 main（被 import 不會）
    main()                                                  # 進入命令列主程式
