import customtkinter as ctk
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageTk
import math
import time
import os
import threading
import json
import tkinter as tk # 引入標準 tkinter 用於 Canvas
from datetime import datetime
import ctypes # 用於解決 Windows DPI 偏移問題

# --- 1. 解決 Windows DPI 縮放導致的座標錯位 ---
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

# --- 設定外觀 ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

CONFIG_FILE = "config.json"

class PCBMeasurementApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- 設定主視窗 ---
        self.title("PCB AI Auto-Measurement System (V4-Precision)")
        self.geometry("1600x900")
        self.resizable(False, False)
        
        # --- 預設參數 ---
        self.params = {
            "mm_per_pixel": 0.05,
            "mask_redundancy": 0,
            "exposure": -5,
            "brightness": 128
        }
        self.load_config()

        self.model_path = "pcb_poc_v2_best.pt" 
        self.camera_index = 0  # 根據您的電腦調整，通常是 0 或 1
        self.conf_threshold = 0.5
        
        # [修改點 1] 冷卻時間變數與預設值
        self.default_cooldown = 3.0 # 標準模式
        self.rapid_cooldown = 0.5   # 急速模式
        self.cooldown_seconds = self.default_cooldown 
        
        self.last_capture_time = 0
        self.is_running = True
        self.save_dir = "measure_results"
        
        # 校正模式變數
        self.is_calibrating = False
        self.calib_points = [] 
        self.current_ratio = 1.0
        
        # 定義顯示區域的固定尺寸
        self.view_width = 640  
        self.view_height = 800 

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # --- 載入模型 ---
        self.status_var = ctk.StringVar(value="正在載入模型...")
        try:
            self.model = YOLO(self.model_path) 
            self.status_var.set("模型載入完成 | 等待偵測...")
        except Exception as e:
            self.status_var.set(f"模型載入失敗: {e}")
            print(f"Error loading model: {e}")

        # --- 建立 UI ---
        self.setup_ui()

        # --- 啟動攝影機 ---
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        self.set_exposure(self.params["exposure"])
        self.set_brightness(self.params["brightness"])

        # --- 啟動影像更新迴圈 ---
        self.update_camera()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    loaded = json.load(f)
                    self.params.update(loaded)
            except:
                pass

    def save_config(self):
        self.params["mm_per_pixel"] = float(self.entry_ratio.get())
        self.params["mask_redundancy"] = int(self.slider_redundancy.get())
        self.params["exposure"] = int(self.slider_exposure.get())
        self.params["brightness"] = int(self.slider_brightness.get())
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.params, f, indent=4)

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=0)
        self.grid_columnconfigure(2, weight=0)
        self.grid_rowconfigure(0, weight=1)

        # ==================== 左側控制欄 ====================
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        
        ctk.CTkLabel(self.sidebar, text="控制面板 (V4)", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=20)

        # --- 校正區塊 ---
        calib_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        calib_frame.pack(pady=10, padx=20, fill="x")
        
        ctk.CTkLabel(calib_frame, text="像素校正 (mm/pixel):", anchor="w").pack(fill="x")
        self.entry_ratio = ctk.CTkEntry(calib_frame)
        self.entry_ratio.insert(0, str(self.params["mm_per_pixel"]))
        self.entry_ratio.pack(pady=5, fill="x")

        self.btn_calib = ctk.CTkButton(calib_frame, text="📏 啟動兩點校正", command=self.toggle_calibration, fg_color="#D35B58")
        self.btn_calib.pack(pady=5, fill="x")
        ctk.CTkLabel(calib_frame, text="* 點擊兩點後輸入長度", font=("Arial", 10), text_color="gray").pack()

        # --- 其他參數 ---
        ctk.CTkLabel(self.sidebar, text="Mask 膨脹 (px):").pack(pady=(10, 0), padx=20, anchor="w")
        self.slider_redundancy = ctk.CTkSlider(self.sidebar, from_=0, to=20, number_of_steps=20)
        self.slider_redundancy.set(self.params["mask_redundancy"])
        self.slider_redundancy.pack(pady=5, padx=20, fill="x")

        ctk.CTkLabel(self.sidebar, text="曝光 (Exposure):").pack(pady=(20, 0), padx=20, anchor="w")
        self.slider_exposure = ctk.CTkSlider(self.sidebar, from_=-13, to=0, command=self.set_exposure)
        self.slider_exposure.set(self.params["exposure"])
        self.slider_exposure.pack(pady=5, padx=20, fill="x")

        ctk.CTkLabel(self.sidebar, text="亮度 (Brightness):").pack(pady=(10, 0), padx=20, anchor="w")
        self.slider_brightness = ctk.CTkSlider(self.sidebar, from_=0, to=255, command=self.set_brightness)
        self.slider_brightness.set(self.params["brightness"])
        self.slider_brightness.pack(pady=5, padx=20, fill="x")
        
        # [修改點 2] 新增「急速偵測」開關 (Switch)
        # 這會切換 self.cooldown_seconds 的值
        self.switch_rapid = ctk.CTkSwitch(self.sidebar, text="⚡ 急速偵測模式 (無延遲)", command=self.toggle_rapid_mode, onvalue=1, offvalue=0)
        self.switch_rapid.pack(pady=(20, 5), padx=20, fill="x")
        
        # 手動按鈕
        self.btn_manual = ctk.CTkButton(self.sidebar, text="📸 手動觸發量測", command=self.manual_trigger, fg_color="green")
        self.btn_manual.pack(pady=10, padx=20, fill="x")

        self.lbl_status = ctk.CTkLabel(self.sidebar, textvariable=self.status_var, wraplength=240)
        self.lbl_status.pack(pady=20, padx=20, side="bottom")

        # ==================== 中間：即時影像 (Canvas) ====================
        self.frame_live_container = ctk.CTkFrame(self)
        self.frame_live_container.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(self.frame_live_container, text="即時影像 (點擊以校正)", font=ctk.CTkFont(size=16)).pack(pady=5)
        
        self.canvas_live = tk.Canvas(self.frame_live_container, width=self.view_width, height=self.view_height, bg="#2b2b2b", highlightthickness=0)
        self.canvas_live.pack(padx=5, pady=5)
        self.canvas_live.bind("<Button-1>", self.on_canvas_click)

        # ==================== 右側：量測結果 ====================
        self.frame_result_container = ctk.CTkFrame(self)
        self.frame_result_container.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(self.frame_result_container, text="最新量測結果", font=ctk.CTkFont(size=16)).pack(pady=5)

        self.canvas_result = tk.Canvas(self.frame_result_container, width=self.view_width, height=self.view_height, bg="#2b2b2b", highlightthickness=0)
        self.canvas_result.pack(padx=5, pady=5)

    # [修改點 3] 切換急速模式的邏輯函式
    def toggle_rapid_mode(self):
        if self.switch_rapid.get() == 1:
            self.cooldown_seconds = self.rapid_cooldown # 設定為 0.5 秒
            self.status_var.set("🚀 急速模式已開啟：偵測到即刻量測")
        else:
            self.cooldown_seconds = self.default_cooldown # 設定為 3.0 秒
            self.status_var.set("🐢 標準模式：等待 3 秒防手震")

    def toggle_calibration(self):
        self.is_calibrating = not self.is_calibrating
        if self.is_calibrating:
            self.calib_points = []
            self.status_var.set("📏 校正模式: 請點擊兩個參考點")
            self.btn_calib.configure(text="❌ 取消校正", fg_color="gray")
        else:
            self.status_var.set("已取消校正")
            self.btn_calib.configure(text="📏 啟動兩點校正", fg_color="#D35B58")

    def on_canvas_click(self, event):
        if not self.is_calibrating:
            return
        ui_x, ui_y = event.x, event.y
        real_img_x = ui_x / self.current_ratio
        real_img_y = ui_y / self.current_ratio
        self.calib_points.append((real_img_x, real_img_y))
        
        if len(self.calib_points) == 1:
            self.status_var.set("已記錄第一點，請點擊第二點...")
        elif len(self.calib_points) == 2:
            self.finish_calibration()

    def finish_calibration(self):
        p1 = np.array(self.calib_points[0])
        p2 = np.array(self.calib_points[1])
        dist_px = np.linalg.norm(p1 - p2)

        if dist_px < 5:
            self.status_var.set("❌ 距離太近，請重新操作")
            self.calib_points = []
            return

        dialog = ctk.CTkInputDialog(text=f"兩點像素距離: {dist_px:.1f} px\n請輸入這段距離的真實長度 (mm):", title="輸入真實長度")
        input_str = dialog.get_input()
        
        if input_str:
            try:
                real_mm = float(input_str)
                new_ratio = real_mm / dist_px
                self.entry_ratio.delete(0, "end")
                self.entry_ratio.insert(0, f"{new_ratio:.5f}")
                self.save_config()
                self.status_var.set(f"✅ 校正成功! {new_ratio:.5f} mm/px")
            except ValueError:
                self.status_var.set("❌ 輸入格式錯誤")
        
        self.is_calibrating = False
        self.btn_calib.configure(text="📏 啟動兩點校正", fg_color="#D35B58")
        self.calib_points = []

    def set_exposure(self, value):
        self.cap.set(cv2.CAP_PROP_EXPOSURE, int(value))

    def set_brightness(self, value):
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, int(value))

    def update_camera(self):
        if not self.is_running: return

        ret, frame = self.cap.read()
        if ret:
            display_frame = frame.copy()
            
            if self.is_calibrating:
                cv2.putText(display_frame, "CALIBRATION MODE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                for pt in self.calib_points:
                    cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
                if len(self.calib_points) == 2:
                    p1 = (int(self.calib_points[0][0]), int(self.calib_points[0][1]))
                    p2 = (int(self.calib_points[1][0]), int(self.calib_points[1][1]))
                    cv2.line(display_frame, p1, p2, (255, 0, 0), 2)
            else:
                results = self.model(frame, verbose=False, conf=self.conf_threshold)
                if results[0].boxes:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # [核心邏輯] 這裡會根據 switch 的狀態決定使用 3.0秒 還是 0.5秒
                    current_time = time.time()
                    if (current_time - self.last_capture_time) > self.cooldown_seconds:
                        self.last_capture_time = current_time
                        self.status_var.set("⚡ 偵測觸發！正在量測...")
                        threading.Thread(target=self.process_measurement, args=(frame.copy(), results[0])).start()

            img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            orig_w, orig_h = img_pil.size
            ratio = min(self.view_width / orig_w, self.view_height / orig_h)
            new_w = int(orig_w * ratio)
            new_h = int(orig_h * ratio)
            self.current_ratio = ratio 
            img_pil_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.photo_image = ImageTk.PhotoImage(img_pil_resized) 
            self.canvas_live.delete("all") 
            self.canvas_live.create_image(0, 0, image=self.photo_image, anchor='nw')

        self.after(30, self.update_camera)

    def manual_trigger(self):
        ret, frame = self.cap.read()
        if ret:
            self.status_var.set("📸 手動觸發量測...")
            results = self.model(frame, verbose=False)
            threading.Thread(target=self.process_measurement, args=(frame.copy(), results[0])).start()

    # --- 演算法 ---
    def detect_edge_slice_fit_raw(self, img, slices=20):
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)).mean()
        sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)).mean()
        mode = 'vertical' if sobel_x > sobel_y else 'horizontal'
        edge_points = []
        step = h // slices if mode == 'vertical' else w // slices
        for i in range(slices):
            start = i * step
            end = (i + 1) * step
            if end > (h if mode == 'vertical' else w): end = (h if mode == 'vertical' else w)
            if mode == 'vertical':
                strip = gray[start:end, :]
                profile = np.mean(strip, axis=0)
                center_pos = (start + end) // 2
            else:
                strip = gray[:, start:end]
                profile = np.mean(strip, axis=1)
                center_pos = (start + end) // 2
            gradient = np.abs(np.gradient(profile))
            if len(gradient) > 10:
                margin = int(len(gradient) * 0.05)
                gradient[:margin] = 0; gradient[-margin:] = 0
            edge_idx = np.argmax(gradient)
            max_val = gradient[edge_idx]
            if max_val > 5:
                if mode == 'vertical': edge_points.append((float(edge_idx), float(center_pos)))
                else: edge_points.append((float(center_pos), float(edge_idx)))
        if len(edge_points) < 3: return mode, (0.0, 1.0, w/2.0, h/2.0)
        pts = np.array(edge_points)
        target_vals = pts[:, 0] if mode == 'vertical' else pts[:, 1]
        median_val = np.median(target_vals)
        mask = np.abs(target_vals - median_val) < 50
        clean_pts = pts[mask]
        if len(clean_pts) < 3: clean_pts = pts
        [vx, vy, x0, y0] = cv2.fitLine(clean_pts, cv2.DIST_L2, 0, 0.01, 0.01)
        return mode, (float(vx), float(vy), float(x0), float(y0))

    def process_measurement(self, original_img, ai_result):
        try:
            h, w = original_img.shape[:2]
            display_img = original_img.copy()
            try: mm_per_pixel = float(self.entry_ratio.get())
            except: mm_per_pixel = 0.05
            redundancy_px = int(self.slider_redundancy.get())
            mode, fit_params = self.detect_edge_slice_fit_raw(original_img)
            edge_vx, edge_vy, edge_x0, edge_y0 = fit_params
            scale = max(h, w) * 2
            p1_edge = (int(edge_x0 - edge_vx * scale), int(edge_y0 - edge_vy * scale))
            p2_edge = (int(edge_x0 + edge_vx * scale), int(edge_y0 + edge_vy * scale))
            cv2.line(display_img, p1_edge, p2_edge, (0, 0, 255), 3)
            raw_mask = np.zeros((h, w), dtype=np.uint8)
            found_mask = False
            if ai_result.masks:
                contour_points = ai_result.masks.xy[0].astype(np.int32)
                cv2.fillPoly(raw_mask, [contour_points], 1)
                found_mask = True
            if found_mask:
                if redundancy_px > 0:
                    kernel = np.ones((redundancy_px * 2 + 1, redundancy_px * 2 + 1), np.uint8)
                    raw_mask = cv2.dilate(raw_mask, kernel, iterations=1)
                y_idxs, x_idxs = np.where(raw_mask > 0)
                if len(x_idxs) > 0:
                    mask_points = np.column_stack((x_idxs, y_idxs)).astype(np.float32)
                    P0 = np.array([edge_x0, edge_y0])
                    V_line = np.array([edge_vx, edge_vy])
                    V_normal = np.array([-edge_vy, edge_vx])
                    Vectors_P0_to_Mask = mask_points - P0
                    dists = np.abs(np.dot(Vectors_P0_to_Mask, V_normal))
                    max_idx = np.argmax(dists)
                    best_point = mask_points[max_idx]
                    max_dist_px = dists[max_idx]
                    dist_mm = max_dist_px * mm_per_pixel
                    bx, by = int(best_point[0]), int(best_point[1])
                    proj_len = np.dot(best_point - P0, V_line)
                    foot_point = P0 + proj_len * V_line
                    fx, fy = int(foot_point[0]), int(foot_point[1])
                    cv2.line(display_img, (bx, by), (fx, fy), (0, 255, 255), 2)
                    cv2.circle(display_img, (bx, by), 5, (0, 255, 0), -1)
                    text = f"{dist_mm:.3f} mm"
                    cv2.putText(display_img, text, (bx - 100, by - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                    contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(display_img, contours, -1, (0, 255, 0), 1)
                    self.status_var.set(f"✅ 量測完成: {dist_mm:.3f} mm")
            else:
                self.status_var.set("⚠️ 無有效 Mask")
            self.save_config()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.save_dir}/result_{timestamp}.jpg"
            cv2.imwrite(filename, display_img)
            img_rgb_res = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            img_pil_res = Image.fromarray(img_rgb_res)
            orig_w, orig_h = img_pil_res.size
            ratio = min(self.view_width / orig_w, self.view_height / orig_h)
            n_w = int(orig_w * ratio)
            n_h = int(orig_h * ratio)
            img_res_resized = img_pil_res.resize((n_w, n_h), Image.Resampling.LANCZOS)
            self.res_photo = ImageTk.PhotoImage(img_res_resized)
            def update_res_canvas():
                self.canvas_result.delete("all")
                self.canvas_result.create_image(0, 0, image=self.res_photo, anchor='nw')
            self.after(0, update_res_canvas)
        except Exception as e:
            print(f"Process Error: {e}")
            self.status_var.set(f"處理錯誤: {e}")

    def on_closing(self):
        self.save_config()
        self.is_running = False
        if self.cap.isOpened(): self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = PCBMeasurementApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
