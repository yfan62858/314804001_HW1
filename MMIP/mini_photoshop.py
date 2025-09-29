#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# mini_photoshop.py — Tiny GUI (stdlib tkinter) with draggable log/gamma curves
# 需與 ImageReading.py（你的 a/b/c 演算法）放在同資料夾

import os
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# 匯入你的演算法
import ImageReading as IR


# ---------- IR.Image -> Tk PhotoImage（用臨時 PGM，最穩定） ----------
def image_to_photo(img: IR.Image, max_w=640, max_h=640) -> tk.PhotoImage:
    w, h = img.w, img.h
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        prev = IR.resize_bilinear(img, new_w, new_h)
    else:
        prev = img

    header = f"P5\n{prev.w} {prev.h}\n255\n".encode("ascii")
    raw = bytes(prev.pix)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pgm") as tmp:
        tmp.write(header); tmp.write(raw)
        name = tmp.name
    try:
        photo = tk.PhotoImage(file=name)
    finally:
        try: os.unlink(name)
        except OSError: pass
    return photo


# -# ------------------------- 曲線視窗（log / gamma，可直接拖曳；無套用按鈕） -------------------------
class CurveDialog(tk.Toplevel):
    PAD = 10
    SIZE = 256

    def __init__(self, master, apply_callback, base_image: IR.Image, init_mode="log", init_c=45.0, init_g=0.6):
        """
        apply_callback(lut_bytes, base_image) -> None
          由主視窗提供：把 LUT 作用在 base_image 後顯示（避免累積）
        base_image:
          開窗當下的影像快照（作為預覽與更新的固定基準）
        """
        super().__init__(master)
        self.title("曲線 (log / gamma)")
        self.resizable(False, False)

        self.apply_callback = apply_callback
        self.base_image = base_image
        self.mode = tk.StringVar(value=init_mode)   # "log" or "gamma"
        self.preview = tk.BooleanVar(value=True)   # 預設關閉預覽
        self.c_val = tk.DoubleVar(value=init_c)
        self.g_val = tk.DoubleVar(value=init_g)

        self._first_render = True  # 第一次 redraw 不預覽
        self._armed = False        # 使用者互動後才允許預覽

        # 介面
        body = tk.Frame(self, padx=10, pady=10)
        body.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(body, width=self.SIZE + 2*self.PAD,
                                height=self.SIZE + 2*self.PAD,
                                bg="white", highlightthickness=1,
                                highlightbackground="#ccc", cursor="crosshair")
        self.canvas.grid(row=0, column=0, rowspan=7, padx=(0, 10))

        # 模式切換
        tk.Label(body, text="Mode").grid(row=0, column=1, sticky="w")
        mrow = tk.Frame(body); mrow.grid(row=1, column=1, sticky="w", pady=(0, 8))
        tk.Radiobutton(mrow, text="log",   variable=self.mode, value="log",
               command=self.on_mode_change).pack(side="left")
        tk.Radiobutton(mrow, text="gamma", variable=self.mode, value="gamma",
               command=self.on_mode_change).pack(side="left")

        # log c
        self.row_c = tk.Frame(body); self.row_c.grid(row=2, column=1, sticky="w", pady=2)
        tk.Label(self.row_c, text="c =").pack(side="left")
        
        tk.Spinbox(self.row_c, from_=1, to=200, increment=1, width=7,
           textvariable=self.c_val, command=self.on_param_change).pack(side="left", padx=4)

        # gamma γ
        self.row_g = tk.Frame(body); self.row_g.grid(row=3, column=1, sticky="w", pady=2)
        tk.Label(self.row_g, text="γ =").pack(side="left")
        tk.Spinbox(self.row_g, from_=0.1, to=3.0, increment=0.1, width=7,
           textvariable=self.g_val, command=self.on_param_change).pack(side="left", padx=4)

        # 只保留「關閉」按鈕
        btns = tk.Frame(body); btns.grid(row=6, column=1, sticky="ew", pady=(8,0))
        tk.Button(btns, text="Close", command=self.destroy).pack(side="left")

        # 滑鼠事件：點擊/拖曳 → 反推參數
        self.canvas.bind("<Button-1>", self.on_drag_param)
        self.canvas.bind("<B1-Motion>", self.on_drag_param)

        self.on_mode_change()  # 只重畫，不動圖
    def on_mode_change(self):
        """切換 log/gamma：只更新可見的參數列與曲線，不解鎖、不預覽。"""
        if self.mode.get() == "log":
            self.row_c.grid(); self.row_g.grid_remove()
        else:
            self.row_c.grid_remove(); self.row_g.grid()
        self.redraw()  # 只重畫，不觸發預覽（預覽邏輯在 redraw 的條件中）

    def on_param_change(self):
        """使用者改了 c 或 γ：解鎖預覽，允許同步更新影像。"""
        self._armed = True
        self.redraw()

    # 目前 LUT
    def current_lut(self) -> bytes:
        return IR.lut_log(float(self.c_val.get())) if self.mode.get() == "log" \
               else IR.lut_gamma(float(self.g_val.get()))

    # 重畫曲線與（必要時）預覽
    def redraw(self):
        c = self.canvas
        c.delete("all")
        S, pad = self.SIZE, self.PAD

        # 格線與框
        for t in range(0, 257, 64):
            x = pad + t; y = pad + (S - t)
            c.create_line(pad, y, pad+S, y, fill="#eee")
            c.create_line(x, pad, x, pad+S, fill="#eee")
        c.create_rectangle(pad, pad, pad+S, pad+S, outline="#999")
        c.create_line(pad, pad+S, pad+S, pad, fill="#bbb", dash=(2,2))
        c.create_text(pad+S//2, pad+S+28, text="input (r)", fill="#666", font=("Segoe UI", 9))
        c.create_text(pad-28, pad+S//2, text="output (s)", fill="#666", font=("Segoe UI", 9), angle=90)

        # 曲線
        lut = self.current_lut()
        pts = [(pad+r, pad + (S - lut[r])) for r in range(256)]
        for i in range(255):
            c.create_line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1], fill="#1976d2", width=2)

        # 即時預覽（需：非第一次、已解鎖、勾選預覽）
        if (not self._first_render) and self._armed and self.preview.get() and (self.base_image is not None):
            try:
                self.apply_callback(lut, self.base_image)
            except Exception:
                pass

        self._first_render = False

    # 滑鼠拖曳：反推參數（c 或 γ）
    def on_drag_param(self, event):
        S, pad = self.SIZE, self.PAD
        r = int(round(max(0, min(255, event.x - pad))))
        s = int(round(max(0, min(255, S - (event.y - pad)))))

        import math
        if self.mode.get() == "log":
            denom = math.log1p(max(0, r))
            if denom > 0:
                c_new = max(1.0, min(200.0, s / denom))
                self.c_val.set(round(c_new, 3))
        else:
            rr = max(1, min(254, r)) / 255.0
            ss = max(1, min(254, s)) / 255.0
            denom = math.log(rr)
            if denom != 0:
                g_new = max(0.1, min(3.0, math.log(ss) / denom))
                self.g_val.set(round(g_new, 3))

        self._armed = True
        self.redraw()


# ------------------------------- GUI 主體 -------------------------------
class MiniPS(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mini Photoshop (stdlib only)")
        self.geometry("980x720")

        self.path = None
        self.img_original: IR.Image | None = None
        self.img_current: IR.Image | None = None

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.panel = tk.Frame(self, padx=10, pady=10)
        self.panel.grid(row=0, column=0, sticky="ns")
        self.panel.columnconfigure(0, weight=1)

        self.view = tk.Label(self, bd=1, relief="sunken", bg="#111")
        self.view.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self._build_controls()
        self._update_buttons_state(disabled=True)

    def _build_controls(self):
        # 檔案
        tk.Label(self.panel, text="File", font=("Segoe UI", 10, "bold")).grid(sticky="w")
        tk.Button(self.panel, text="Open (BMP/RAW)", command=self.open_file).grid(sticky="ew", pady=(2, 6))
        tk.Button(self.panel, text="Save as BMP", command=self.save_bmp).grid(sticky="ew")

        ttk.Separator(self.panel, orient="horizontal").grid(sticky="ew", pady=10)

        # (b) 灰階強度轉換（按鈕直接套一次）
        tk.Label(self.panel, text="Image enhancement tool", font=("Segoe UI", 10, "bold")).grid(sticky="w")
        tk.Button(self.panel, text="Negative", command=lambda: self.apply_xform("negative")).grid(sticky="ew", pady=2)

        row2 = tk.Frame(self.panel); row2.grid(sticky="ew", pady=2)
        tk.Label(row2, text="log c=").grid(row=0, column=0)
        self.var_c = tk.DoubleVar(value=45.0)
        tk.Spinbox(row2, from_=1, to=200, increment=1, textvariable=self.var_c, width=7).grid(row=0, column=1, padx=4)
        tk.Button(row2, text="Apply log", command=lambda: self.apply_xform("log")).grid(row=0, column=2)

        row3 = tk.Frame(self.panel); row3.grid(sticky="ew", pady=2)
        tk.Label(row3, text="gamma=").grid(row=0, column=0)
        self.var_g = tk.DoubleVar(value=0.6)
        tk.Spinbox(row3, from_=0.1, to=3.0, increment=0.1, textvariable=self.var_g, width=7).grid(row=0, column=1, padx=4)
        tk.Button(row3, text="Apply gamma", command=lambda: self.apply_xform("gamma")).grid(row=0, column=2)

        # 曲線視窗（可拖曳）
        tk.Button(self.panel, text="Curve(log/gamma)…", command=self.open_curve_dialog).grid(sticky="ew", pady=(6, 2))

        ttk.Separator(self.panel, orient="horizontal").grid(sticky="ew", pady=10)

        # (c) 縮放
        tk.Label(self.panel, text="Image sampling", font=("Segoe UI", 10, "bold")).grid(sticky="w")
        row4 = tk.Frame(self.panel); row4.grid(sticky="ew", pady=2)
        tk.Label(row4, text="W:").grid(row=0, column=0)
        self.var_w = tk.IntVar(value=1024)
        tk.Entry(row4, textvariable=self.var_w, width=7).grid(row=0, column=1, padx=4)
        tk.Label(row4, text="H:").grid(row=0, column=2)
        self.var_h = tk.IntVar(value=1024)
        tk.Entry(row4, textvariable=self.var_h, width=7).grid(row=0, column=3, padx=4)
        tk.Label(row4, text="Method:").grid(row=0, column=4)
        self.var_m = tk.StringVar(value="nearest")
        ttk.Combobox(row4, textvariable=self.var_m, values=["nearest", "bilinear"],
                     width=8, state="readonly").grid(row=0, column=5, padx=4)
        tk.Button(row4, text="Resize", command=self.apply_resize).grid(row=0, column=6, padx=4)

        ttk.Separator(self.panel, orient="horizontal").grid(sticky="ew", pady=10)

        # 其他
        tk.Button(self.panel, text="Reset", command=self.reset_image).grid(sticky="ew", pady=2)

    # ---- 檔案 ----
    def open_file(self):
        path = filedialog.askopenfilename(
            title="選取影像",
            filetypes=[("Images", "*.bmp;*.raw"), ("BMP", "*.bmp"), ("RAW 512x512", "*.raw"), ("All files", "*.*")]
        )
        if not path: return
        try:
            img = IR.load_auto(path)
        except Exception as e:
            messagebox.showerror("讀取失敗", str(e)); return
        self.path = path
        self.img_original = img
        self.img_current = IR.Image(img.w, img.h, bytearray(img.pix))
        self.render()

    def save_bmp(self):
        if not self.img_current: return
        out = filedialog.asksaveasfilename(defaultextension=".bmp",
                                           filetypes=[("BMP (24-bit grayscale)", "*.bmp")])
        if not out: return
        try:
            IR.save_bmp24_gray(self.img_current, out)
            messagebox.showinfo("Finish", f"Saved：\n{out}")
        except Exception as e:
            messagebox.showerror("儲存失敗", str(e))

    # ---- 影像處理（按鈕一次套用）----
    def apply_xform(self, typ: str):
        if not self.img_current: return
        try:
            if typ == "negative":
                self.img_current = IR.apply_lut(self.img_current, IR.lut_negative())
            elif typ == "log":
                self.img_current = IR.apply_lut(self.img_current, IR.lut_log(float(self.var_c.get())))
            elif typ == "gamma":
                self.img_current = IR.apply_lut(self.img_current, IR.lut_gamma(float(self.var_g.get())))
            self.render()
        except Exception as e:
            messagebox.showerror("轉換失敗", str(e))

    def apply_resize(self):
        if not self.img_current: return
        try:
            w = int(self.var_w.get()); h = int(self.var_h.get())
            m = self.var_m.get()
            if w <= 0 or h <= 0: raise ValueError("寬高需為正整數")
            self.img_current = IR.resize_nearest(self.img_current, w, h) if m == "nearest" \
                               else IR.resize_bilinear(self.img_current, w, h)
            self.render()
        except Exception as e:
            messagebox.showerror("縮放失敗", str(e))

    def reset_image(self):
        if not self.img_original: return
        self.img_current = IR.Image(self.img_original.w, self.img_original.h,
                                    bytearray(self.img_original.pix))
        self.render()

    # ---- 曲線視窗（可拖曳）----
    def open_curve_dialog(self):
        if not self.img_current:
            messagebox.showinfo("提示", "請先開啟一張影像。")
            return

        # 在開窗瞬間拍一張基準（避免預覽時累積）
        base = IR.Image(self.img_current.w, self.img_current.h, bytearray(self.img_current.pix))

        def apply_from_curve(lut_bytes: bytes, base_image: IR.Image):
            # 每次以 base_image 做映射，避免累積
            self.img_current = IR.apply_lut(base_image, lut_bytes)
            self.render()
            # 同步把外層 Spinbox 也更新，保持一致
            if dlg.mode.get() == "log":
                self.var_c.set(dlg.c_val.get())
            else:
                self.var_g.set(dlg.g_val.get())

        dlg = CurveDialog(self, apply_callback=apply_from_curve, base_image=base,
                          init_mode="log", init_c=float(self.var_c.get()), init_g=float(self.var_g.get()))
    def render(self):
        if not self.img_current: return
        photo = image_to_photo(self.img_current, max_w=900, max_h=680)
        self.view.configure(image=photo); self.view.image = photo
        self._update_buttons_state(disabled=False)
        name = os.path.basename(self.path) if self.path else "(未命名)"
        self.title(f"Mini Photoshop — {name}  [{self.img_current.w}x{self.img_current.h}]")

    def _update_buttons_state(self, disabled: bool):
        for child in self.panel.winfo_children():
            if isinstance(child, (tk.Button, ttk.Combobox, tk.Spinbox, tk.Entry)):
                try: child.configure(state=("disabled" if disabled else "normal"))
                except tk.TclError: pass
        self.panel.winfo_children()[1].configure(state="normal")  # 開啟檔案永遠可用


if __name__ == "__main__":
    app = MiniPS()
    app.mainloop()
