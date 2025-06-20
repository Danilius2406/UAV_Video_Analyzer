import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import os

class TestVideoGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("Test Video Generator")
        self.setup_ui()

    def setup_ui(self):
        param_frame = ttk.LabelFrame(self.root, text="Video Parameters")
        param_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(param_frame, text="Resolution (WxH):").grid(row=0, column=0)
        self.res_w = ttk.Entry(param_frame, width=8)
        self.res_w.insert(0, "1920")
        self.res_w.grid(row=0, column=1)
        self.res_h = ttk.Entry(param_frame, width=8)
        self.res_h.insert(0, "1080")
        self.res_h.grid(row=0, column=2)

        ttk.Label(param_frame, text="FPS:").grid(row=1, column=0)
        self.fps = ttk.Entry(param_frame, width=8)
        self.fps.insert(0, "60")
        self.fps.grid(row=1, column=1)

        ttk.Label(param_frame, text="Duration (sec):").grid(row=2, column=0)
        self.duration = ttk.Entry(param_frame, width=8)
        self.duration.insert(0, "5")
        self.duration.grid(row=2, column=1)

        motion_frame = ttk.LabelFrame(self.root, text="Motion Parameters")
        motion_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(motion_frame, text="Motion Type:").grid(row=0, column=0)
        self.motion_type = ttk.Combobox(motion_frame,
                                        values=["Constant Velocity", "Accelerated", "Sinusoidal"])
        self.motion_type.current(0)
        self.motion_type.grid(row=0, column=1)

        ttk.Label(motion_frame, text="Initial Velocity (m/s):").grid(row=1, column=0)
        self.init_velocity = ttk.Entry(motion_frame, width=8)
        self.init_velocity.insert(0, "10")
        self.init_velocity.grid(row=1, column=1)

        ttk.Label(motion_frame, text="Acceleration (m/s²):").grid(row=2, column=0)
        self.acceleration = ttk.Entry(motion_frame, width=8)
        self.acceleration.insert(0, "0")
        self.acceleration.grid(row=2, column=1)

        noise_frame = ttk.LabelFrame(self.root, text="Noise Parameters")
        noise_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(noise_frame, text="Motion Blur:").grid(row=0, column=0)
        self.motion_blur = ttk.Combobox(noise_frame,
                                        values=["None", "Low", "Medium", "High"])
        self.motion_blur.current(0)
        self.motion_blur.grid(row=0, column=1)

        ttk.Label(noise_frame, text="Occlusions:").grid(row=1, column=0)
        self.occlusions = ttk.Combobox(noise_frame,
                                       values=["None", "Partial", "Full"])
        self.occlusions.current(0)
        self.occlusions.grid(row=1, column=1)

        btn_frame = ttk.Frame(self.root)
        btn_frame.grid(row=3, column=0, pady=10)

        ttk.Button(btn_frame, text="Generate Video",
                   command=self.generate_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Exit",
                   command=self.root.quit).pack(side=tk.LEFT, padx=5)

        self.progress = ttk.Progressbar(self.root, orient=tk.HORIZONTAL,
                                        mode='determinate')
        self.progress.grid(row=4, column=0, padx=10, pady=5, sticky="ew")

    def generate_video(self):
        width = int(self.res_w.get())
        height = int(self.res_h.get())
        fps = int(self.fps.get())
        duration = int(self.duration.get())
        total_frames = fps * duration

        motion_type = self.motion_type.get()
        v0 = float(self.init_velocity.get())
        a = float(self.acceleration.get())

        scale = 0.01
        v0_px = v0 / scale * (1/fps)
        a_px = a / scale * (1/fps**2)

        x = np.zeros(total_frames)
        y = np.zeros(total_frames)

        if motion_type == "Constant Velocity":
            x = np.linspace(100, 100 + v0_px*total_frames, total_frames)
            y = np.full(total_frames, height//2)

        elif motion_type == "Accelerated":
            t = np.arange(total_frames)
            x = 100 + v0_px*t + 0.5*a_px*t**2
            y = np.full(total_frames, height//2)

        elif motion_type == "Sinusoidal":
            t = np.arange(total_frames)
            x = 100 + v0_px*t
            y = height//2 + 100*np.sin(0.1*t)

        save_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])

        if not save_path:
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        ground_truth = []

        for i in range(total_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            cv2.line(frame, (50, height-50), (50 + int(1/scale), height-50),
                     (0, 255, 0), 2)

            obj_size = 20
            pt1 = (int(x[i]-obj_size//2), int(y[i]-obj_size//2))
            pt2 = (int(x[i]+obj_size//2), int(y[i]+obj_size//2))
            cv2.rectangle(frame, pt1, pt2, (0, 0, 255), -1)

            blur_level = self.motion_blur.get()
            if blur_level == "Low":
                frame = cv2.GaussianBlur(frame, (3, 3), 0)
            elif blur_level == "Medium":
                frame = cv2.GaussianBlur(frame, (5, 5), 0)
            elif blur_level == "High":
                frame = cv2.GaussianBlur(frame, (7, 7), 0)

            if self.occlusions.get() != "None":
                if i % 50 == 0 and i > 0:
                    cv2.rectangle(frame,
                                  (int(x[i]-50), int(y[i]-50)),
                                  (int(x[i]+50), int(y[i]+50)),
                                  (0, 255, 0), -1)

            out.write(frame)

            ground_truth.append({
                'frame': i,
                'time': i/fps,
                'x_true': x[i],
                'y_true': y[i],
                'vx_true': v0 + a*(i/fps),
                'vy_true': 0.0
            })

            self.progress['value'] = (i+1)/total_frames*100
            self.root.update_idletasks()

        out.release()

        df = pd.DataFrame(ground_truth)
        csv_path = os.path.splitext(save_path)[0] + "_ground_truth.csv"
        df.to_csv(csv_path, index=False)

        tk.messagebox.showinfo("Success",
                               f"Video and ground truth data saved to:\n{save_path}\n{csv_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TestVideoGenerator(root)
    root.mainloop()