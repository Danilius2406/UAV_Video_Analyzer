import cv2
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import (QFileDialog, QVBoxLayout, QPushButton,
                             QLabel, QHBoxLayout, QTableWidget,
                             QTableWidgetItem, QMessageBox, QInputDialog,
                             QSpinBox, QFormLayout)
from PyQt5.QtCore import QTimer, Qt, QPoint
import sys
import traceback
from zoom_handler import ZoomableVideoWidget

class UAVAnalyzer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.paused = True
        self.cap = None
        self.positions = []
        self.timestamps = []
        self.velocities = []
        self.accelerations = []
        self.scale_factor = 1.0
        self.fps = 30
        self.frame_count = 0
        self.tracking_point = None
        self.search_radius = 50
        self.template = None
        self.template_size = 50
        self.tracking_radius = self.template_size // 2
        self.user_declared_fps = 30.0
        self.current_frame = None
        self.tracking_active = False
        self.setFocusPolicy(Qt.StrongFocus)
        self.trajectory = []  # Stores all historical points
        self.show_trajectory = False
        self.trajectory_color = (0, 0, 255)  # red path
        self.trajectory_thickness = 2
        self.dash_length = 10  # Length of each dash segment
        self.gap_length = 5    # Length of gap between dashes
        self.max_trajectory_points = 1000  # Maximum points to keep in trajectory
        self.video_label.clicked.connect(self.handle_video_click)
        self.setting_tracking = False

        self.kalman = cv2.KalmanFilter(4, 2)
        self.init_kalman()
        self.template_alpha = 0.95  # Template update blending factor
        self.min_confidence = 0.5
        self.max_speed = 20 #px/frame
        self.MAX_PHYSICAL_ACCEL = 50.0
        self.MIN_PHYSICAL_ACCEL = -50.0
        self.velocity_filter = None

    def initUI(self):
        self.setWindowTitle("UAV Tracking Analyzer")
        self.setGeometry(100, 100, 1200, 800)

        # Main widgets
        self.video_label = ZoomableVideoWidget(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)

        self.table = QTableWidget(self)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Distance (m)", "Time (s)", "Velocity (m/s)", "Acceleration (m/s²)"])
        self.table.setMinimumWidth(400)

        video_table_layout = QHBoxLayout()
        video_table_layout.addWidget(self.video_label, 70)  # 70% width for video
        video_table_layout.addWidget(self.table, 30)        # 30% width for table

        self.load_button = QPushButton("Load Video", self)
        self.load_button.clicked.connect(self.load_video)

        self.play_button = QPushButton("Play", self)
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.setEnabled(False)

        self.export_button = QPushButton("Export Data", self)
        self.export_button.clicked.connect(self.export_data)
        self.export_button.setEnabled(False)

        self.set_scale_button = QPushButton("Set Scale", self)
        self.set_scale_button.clicked.connect(self.set_scale)
        self.set_scale_button.setEnabled(False)

        self.set_tracking_button = QPushButton("Set Tracking Point", self)
        self.set_tracking_button.clicked.connect(self.toggle_set_tracking_mode)
        self.set_tracking_button.setEnabled(False)

        self.prev_frame_button = QPushButton("Previous Frame", self)
        self.prev_frame_button.clicked.connect(self.prev_frame)
        self.prev_frame_button.setEnabled(False)

        self.next_frame_button = QPushButton("Next Frame", self)
        self.next_frame_button.clicked.connect(self.next_frame)
        self.next_frame_button.setEnabled(False)

        self.check_fps_button = QPushButton("Check FPS", self)
        self.check_fps_button.clicked.connect(self.check_video_fps)
        self.check_fps_button.setEnabled(False)

        self.traj_button = QPushButton("Show Trajectory", self)
        self.traj_button.clicked.connect(self.toggle_trajectory)
        self.traj_button.setEnabled(False)

        self.traj_style_button = QPushButton("Trajectory Style", self)
        self.traj_style_button.clicked.connect(self.set_trajectory_style)
        self.traj_style_button.setEnabled(False)

        self.clear_traj_button = QPushButton("Clear Trajectory", self)
        self.clear_traj_button.clicked.connect(self.clear_trajectory)
        self.clear_traj_button.setEnabled(False)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.prev_frame_button)
        button_layout.addWidget(self.next_frame_button)
        button_layout.addWidget(self.set_scale_button)
        button_layout.addWidget(self.set_tracking_button)
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.check_fps_button)
        button_layout.addWidget(self.traj_button)
        button_layout.addWidget(self.traj_style_button)
        button_layout.addWidget(self.clear_traj_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(button_layout)
        main_layout.addLayout(video_table_layout)

        container = QtWidgets.QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.status_bar = self.statusBar()


    def init_kalman(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)

        self.velocity_filter = cv2.KalmanFilter(2, 1)
        self.velocity_filter.measurementMatrix = np.array([[1, 0]], dtype=np.float32)
        self.velocity_filter.transitionMatrix = np.array([[1, 1], [0, 1]], dtype=np.float32)
        self.velocity_filter.processNoiseCov = np.array([[0.07, 0.0], [0.0, 0.2]], dtype=np.float32)
        self.velocity_filter.measurementNoiseCov = np.eye(1, dtype=np.float32) * 0.3
        self.velocity_filter.errorCovPost = np.eye(2, dtype=np.float32)

        self.accel_filter = cv2.KalmanFilter(1, 1)
        self.accel_filter.measurementMatrix = np.eye(1, dtype=np.float32)
        self.accel_filter.transitionMatrix = np.eye(1, dtype=np.float32)
        self.accel_filter.processNoiseCov = np.eye(1, dtype=np.float32) * 0.1
        self.accel_filter.measurementNoiseCov = np.eye(1, dtype=np.float32) * 10.0
        self.accel_filter.errorCovPost = np.eye(1, dtype=np.float32)

    def handle_video_click(self, pos):
        try:
            if not self.setting_tracking or self.current_frame is None:
                return

            frame_height, frame_width = self.current_frame.shape[:2]
            x = max(0, min(pos.x(), frame_width-1))
            y = max(0, min(pos.y(), frame_height-1))

            self.set_tracking_point(QPoint(x, y))

        except Exception as e:
            self.status_bar.showMessage(f"Click error: {str(e)}")

    def set_tracking_point(self, pos):
        try:
            if self.current_frame is None:
                self.status_bar.showMessage("Error: No video frame loaded")
                return

            frame_height, frame_width = self.current_frame.shape[:2]

            x = max(0, min(pos.x(), frame_width-1))
            y = max(0, min(pos.y(), frame_height-1))

            self.tracking_point = QPoint(x, y)
            self.tracking_active = True

            self.kalman.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.kalman_initialized = True

            frame_gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
            half_size = self.template_size // 2

            x1 = max(0, x - half_size)
            y1 = max(0, y - half_size)
            x2 = min(frame_width, x + half_size)
            y2 = min(frame_height, y + half_size)

            if x1 >= x2 or y1 >= y2:
                raise ValueError("Invalid template region")

            self.template = frame_gray[y1:y2, x1:x2]

            marker_frame = self.current_frame.copy()
            cv2.drawMarker(marker_frame, (x, y), (0, 255, 255),
                           markerType=cv2.MARKER_CROSS,
                           markerSize=20,
                           thickness=2)
            self.display_frame(marker_frame)
            self.status_bar.showMessage(f"Tracking point: ({x}, {y})")

        except Exception as e:
            self.status_bar.showMessage(f"Tracking Error: {str(e)}")
            self.tracking_active = False

    def toggle_set_tracking_mode(self):
        self.setting_tracking = not self.setting_tracking
        self.video_label.allow_pan = not self.setting_tracking

        if self.setting_tracking:
            self.set_tracking_button.setText("Finish Setting")
            self.status_bar.showMessage("Click on the UAV to set tracking point")
            self.video_label.setCursor(Qt.CrossCursor)
        else:
            self.set_tracking_button.setText("Set Tracking Point")
            self.status_bar.showMessage("Tracking point set")
            self.video_label.unsetCursor()
            if self.tracking_point:
                self.update_template_from_point(self.tracking_point)

    def update_template_from_point(self, point):
        if self.tracking_point and self.current_frame is not None:
            x = self.tracking_point.x()
            y = self.tracking_point.y()
            frame_gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
            half_size = self.template_size // 2

            x1 = max(0, x - half_size)
            y1 = max(0, y - half_size)
            x2 = min(frame_gray.shape[1], x + half_size)
            y2 = min(frame_gray.shape[0], y + half_size)

            self.template = frame_gray[y1:y2, x1:x2]

    def set_trajectory_style(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Trajectory Style Settings")

        layout = QFormLayout()

        color_button = QPushButton("Select Color")
        color_button.clicked.connect(lambda: self.select_trajectory_color(dialog))
        layout.addRow("Color:", color_button)

        thickness_spin = QSpinBox()
        thickness_spin.setRange(1, 10)
        thickness_spin.setValue(self.trajectory_thickness)
        thickness_spin.valueChanged.connect(lambda v: setattr(self, 'trajectory_thickness', v))
        layout.addRow("Thickness:", thickness_spin)

        dash_spin = QSpinBox()
        dash_spin.setRange(1, 50)
        dash_spin.setValue(self.dash_length)
        dash_spin.valueChanged.connect(lambda v: setattr(self, 'dash_length', v))
        layout.addRow("Dash Length:", dash_spin)

        gap_spin = QSpinBox()
        gap_spin.setRange(1, 50)
        gap_spin.setValue(self.gap_length)
        gap_spin.valueChanged.connect(lambda v: setattr(self, 'gap_length', v))
        layout.addRow("Gap Length:", gap_spin)

        points_spin = QSpinBox()
        points_spin.setRange(10, 10000)
        points_spin.setValue(self.max_trajectory_points)
        points_spin.valueChanged.connect(lambda v: setattr(self, 'max_trajectory_points', v))
        layout.addRow("Max Points:", points_spin)

        dialog.setLayout(layout)
        dialog.exec_()

        if self.show_trajectory and self.current_frame is not None:
            self.display_frame(self.current_frame)

    def select_trajectory_color(self, parent):
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(*self.trajectory_color), parent, "Select Trajectory Color")
        if color.isValid():
            self.trajectory_color = (color.red(), color.green(), color.blue())
            if self.show_trajectory and self.current_frame is not None:
                self.display_frame(self.current_frame)

    def draw_dashed_line(self, img, pt1, pt2, color, thickness, dash_length, gap_length):
        dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        if dist == 0:
            return

        dx = (pt2[0] - pt1[0]) / dist
        dy = (pt2[1] - pt1[1]) / dist

        drawn = 0
        while drawn < dist:
            start = (int(pt1[0] + dx * drawn), int(pt1[1] + dy * drawn))
            end_dash = min(dist, drawn + dash_length)
            end = (int(pt1[0] + dx * end_dash), int(pt1[1] + dy * end_dash))
            cv2.line(img, start, end, color, thickness)
            drawn = end_dash + gap_length

    def clear_trajectory(self):
        self.trajectory = []
        if self.current_frame is not None:
            self.display_frame(self.current_frame)
        self.tracking_point = None

    def check_video_fps(self):
        if self.cap is None: return

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        meta_fps = self.cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / meta_fps

        msg = (f"Video Analysis:\n"
               f"Metadata FPS: {meta_fps:.2f}\n"
               f"Total frames: {total_frames}\n"
               f"Metadata duration: {duration:.2f}s\n"
               f"User-declared FPS: {self.user_declared_fps:.2f}\n\n"
               )

        QMessageBox.information(self, "FPS Analysis", msg)

    def load_video(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)",
            options=options)

        if file_path:
            self.video_path = file_path
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Could not open video file.")
                return

            self.metadata_fps = self.cap.get(cv2.CAP_PROP_FPS)
            fps, ok = QInputDialog.getDouble(
                self, "Video FPS Settings",
                f"Metadata FPS: {self.metadata_fps:.2f}\nEnter analysis FPS:",
                value=self.metadata_fps,
                min=1.0,
                max=10000.0,
                decimals=2
            )

            if ok:
                self.user_declared_fps = fps
            else:
                self.user_declared_fps = self.metadata_fps

            self.positions = []
            self.timestamps = []
            self.velocities = []
            self.accelerations = []
            self.frame_count = 0
            self.tracking_point = None
            self.template = None
            self.tracking_active = False
            self.trajectory = []

            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_frame)

            self.play_button.setEnabled(True)
            self.set_scale_button.setEnabled(True)
            self.set_tracking_button.setEnabled(True)
            self.prev_frame_button.setEnabled(True)
            self.next_frame_button.setEnabled(True)
            self.check_fps_button.setEnabled(True)
            self.traj_button.setEnabled(True)
            self.traj_style_button.setEnabled(True)
            self.clear_traj_button.setEnabled(True)

            self.status_bar.showMessage(f"Loaded: {file_path} | FPS: {self.user_declared_fps:.2f} - Click on the UAV to start tracking")

            self.show_frame()

    def show_frame(self):
        if self.cap is not None:
            current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                self.display_frame(frame)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

    def display_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.show_trajectory and len(self.trajectory) > 1:
            for i in range(1, len(self.trajectory)):
                self.draw_dashed_line(
                    frame_rgb,
                    (self.trajectory[i-1][0], self.trajectory[i-1][1]),
                    (self.trajectory[i][0], self.trajectory[i][1]),
                    self.trajectory_color,
                    self.trajectory_thickness,
                    self.dash_length,
                    self.gap_length
                )

        if self.tracking_point:
            x = self.tracking_point.x()
            y = self.tracking_point.y()

            cv2.circle(frame_rgb, (x, y),
                       self.tracking_radius,
                       (255, 0, 0),  # red
                       2)

            cv2.rectangle(frame_rgb,
                          (x - self.search_radius, y - self.search_radius),
                          (x + self.search_radius, y + self.search_radius),
                          (0, 255, 0),  # green
                          1)  # thickness

            cv2.drawMarker(frame_rgb,
                           (x, y),
                           (255, 255, 0),  # yellow
                           markerType=cv2.MARKER_CROSS,
                           markerSize=20,
                           thickness=2)

        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line,
                             QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_img)

        self.video_label.setPixmap(pixmap)
        self.video_label.update_display()

    def update_frame(self):
        if self.paused or not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.export_button.setEnabled(True)
            self.status_bar.showMessage("Video processing complete. Ready to export data.")
            return

        self.current_frame = frame.copy()
        self.frame_count += 1
        current_time = self.frame_count / self.user_declared_fps

        if self.tracking_active and self.tracking_point and self.template is not None:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x_prev, y_prev = self.tracking_point.x(), self.tracking_point.y()

            self.trajectory.append((x_prev, y_prev))
            if len(self.trajectory) > self.max_trajectory_points:
                self.trajectory.pop(0)

            try:
                margin = self.search_radius
                x1 = max(0, x_prev - margin)
                y1 = max(0, y_prev - margin)
                x2 = min(frame_gray.shape[1], x_prev + margin)
                y2 = min(frame_gray.shape[0], y_prev + margin)

                if x1 >= x2 or y1 >= y2:
                    raise ValueError("Invalid search area")

                search_area = frame_gray[y1:y2, x1:x2]
                res = cv2.matchTemplate(search_area, self.template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                if max_val < self.min_confidence:
                    raise ValueError(f"Low match confidence: {max_val:.2f}")

                (py, px) = np.unravel_index(np.argmax(res), res.shape[::-1])
                if 0 < px < res.shape[1]-1 and 0 < py < res.shape[0]-1:
                    dx = res[py, px+1] - res[py, px-1]
                    dy = res[py+1, px] - res[py-1, px]
                    dxx = res[py, px+1] + res[py, px-1] - 2*max_val
                    dyy = res[py+1, px] + res[py-1, px] - 2*max_val
                    subpixel_x = px + (dx / (2*dxx)) if dxx != 0 else px
                    subpixel_y = py + (dy / (2*dyy)) if dyy != 0 else py
                else:
                    subpixel_x = px
                    subpixel_y = py

                new_x = x1 + subpixel_x + self.template.shape[1]//2
                new_y = y1 + subpixel_y + self.template.shape[0]//2

                if not hasattr(self, 'kalman_initialized'):
                    self.kalman.statePost = np.array([[new_x], [new_y], [0], [0]], dtype=np.float32)
                    self.kalman_initialized = True

                prediction = self.kalman.predict()
                measured = np.array([[np.float32(new_x)], [np.float32(new_y)]])
                self.kalman.correct(measured)

                filtered_pos = self.kalman.statePost[:2].ravel()
                new_x, new_y = filtered_pos[0], filtered_pos[1]

                if len(self.positions) >= 1:
                    prev_x, prev_y = self.positions[-1]
                    dx = new_x - prev_x
                    dy = new_y - prev_y
                    distance = np.hypot(dx, dy)

                    if distance > self.max_speed:
                        angle = np.arctan2(dy, dx)
                        new_x = prev_x + self.max_speed * np.cos(angle)
                        new_y = prev_y + self.max_speed * np.sin(angle)

                    if not np.isnan(new_x) and not np.isnan(new_y):
                        self.tracking_point = QPoint(int(round(new_x)), int(round(new_y)))
                    else:
                        self.tracking_active = False

                half_size = self.template_size // 2
                x1_t = max(0, int(new_x) - half_size)
                y1_t = max(0, int(new_y) - half_size)
                x2_t = min(frame_gray.shape[1], int(new_x) + half_size)
                y2_t = min(frame_gray.shape[0], int(new_y) + half_size)

                if x1_t < x2_t and y1_t < y2_t:
                    current_patch = frame_gray[y1_t:y2_t, x1_t:x2_t]
                    if current_patch.shape == self.template.shape and self.frame_count > 5:
                        self.template = cv2.addWeighted(
                            self.template, self.template_alpha,
                            current_patch, 1 - self.template_alpha,
                            0
                        )

                if len(self.positions) < self.frame_count:
                    self.positions.append((new_x, new_y))
                    self.timestamps.append(current_time)
                else:
                    self.positions[self.frame_count-1] = (new_x, new_y)
                    self.timestamps[self.frame_count-1] = current_time

            except Exception as e:
                if hasattr(self, 'kalman_initialized'):
                    prediction = self.kalman.predict()
                    new_x, new_y = prediction[0], prediction[1]
                    self.tracking_point = QPoint(int(new_x), int(new_y))
                    self.status_bar.showMessage(f"Tracking warning: {str(e)}, using prediction")
                else:
                    self.tracking_active = False
                    self.status_bar.showMessage(f"Tracking failed: {str(e)}")

        if len(self.positions) >= 2:
            dt = 1.0 / self.user_declared_fps

            # 1. Розрахунок поточної швидкості в пікселях/кадр
            dx_px = self.positions[-1][0] - self.positions[-2][0]
            dy_px = self.positions[-1][1] - self.positions[-2][1]
            speed_px = np.hypot(dx_px, dy_px)

            # 2. Конвертація в м/с
            current_velocity = speed_px * self.scale_factor / dt

            # 3. Фільтрація швидкості
            if self.velocity_filter is not None:
                self.velocity_filter.predict()
                filtered_velocity = self.velocity_filter.correct(
                    np.array([[current_velocity]], dtype=np.float32)
                )[0, 0]
            else:
                filtered_velocity = current_velocity

            # 4. Розрахунок прискорення
            if len(self.velocities) >= 1:
                prev_velocity = self.velocities[-1]
                current_accel = (filtered_velocity - prev_velocity) / dt

                current_accel = np.clip(current_accel,
                                        self.MIN_PHYSICAL_ACCEL,
                                        self.MAX_PHYSICAL_ACCEL)

                # Фільтрація прискорення
                if hasattr(self, 'accel_filter'):
                    self.accel_filter.predict()
                    filtered_accel = self.accel_filter.correct(
                        np.array([[current_accel]], dtype=np.float32)
                    )[0, 0]
                else:
                    filtered_accel = current_accel
            else:
                filtered_accel = 0.0

            # 5. Збереження результатів
            self.velocities.append(filtered_velocity)
            self.accelerations.append(filtered_accel)

        elif len(self.positions) >= 1:
            # Якщо тільки одна позиція - швидкість 0
            self.velocities.append(0.0)
            self.accelerations.append(0.0)

        # Відображення проміжних результатів для налагодження
        if len(self.positions) >= 2:
            print(f"Frame {self.frame_count}:")
            print(f" Speed (px/fr): {speed_px:.2f} | Velocity (m/s): {current_velocity:.2f}")
            print(f" Filtered vel: {filtered_velocity:.2f} | Acceleration: {filtered_accel:.2f}")

        self.display_frame(frame)
        self.update_table()


    def prev_frame(self):
        if self.cap is not None and self.cap.isOpened():
            self.paused = True
            self.play_button.setText("Play")

            current_frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            prev_frame_number = max(0, current_frame_pos - 2)

            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, prev_frame_number)
                ret, frame = self.cap.read()

                if ret:
                    self.current_frame = frame.copy()
                    self.frame_count = prev_frame_number + 1

                    if len(self.positions) > self.frame_count - 1:
                        x = int(round(self.positions[self.frame_count - 1][0]))
                        y = int(round(self.positions[self.frame_count - 1][1]))
                        self.tracking_point = QPoint(x, y)

                        if self.tracking_active and x is not None and y is not None:
                            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            half_size = self.template_size // 2

                            x1 = max(0, x - half_size)
                            y1 = max(0, y - half_size)
                            x2 = min(frame_gray.shape[1], x + half_size)
                            y2 = min(frame_gray.shape[0], y + half_size)

                            if x1 < x2 and y1 < y2:
                                self.template = frame_gray[y1:y2, x1:x2]

                    self.display_frame(frame)
                    self.status_bar.showMessage(f"Frame {self.frame_count}")
                else:
                    self.status_bar.showMessage("Reached start of video")

            except Exception as e:
                self.status_bar.showMessage(f"Error: {str(e)}")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)

    def next_frame(self):
        if self.cap is not None:
            self.paused = True
            self.play_button.setText("Play")
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                self.frame_count += 1

                if len(self.positions) >= self.frame_count:
                    x = int(round(self.positions[self.frame_count - 1][0]))
                    y = int(round(self.positions[self.frame_count - 1][1]))
                    self.tracking_point = QPoint(x, y)

                    if self.tracking_active:
                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        half_size = self.template_size // 2

                        x1 = max(0, x - half_size)
                        y1 = max(0, y - half_size)
                        x2 = min(frame_gray.shape[1], x + half_size)
                        y2 = min(frame_gray.shape[0], y + half_size)

                        if x1 < x2 and y1 < y2:
                            self.template = frame_gray[y1:y2, x1:x2]
                else:
                    self.tracking_point = None
                    self.tracking_active = False

                self.display_frame(frame)
                status_msg = f"Frame {self.frame_count}"
                if self.tracking_point:
                    status_msg += f": Position ({self.tracking_point.x()}, {self.tracking_point.y()})"
                self.status_bar.showMessage(status_msg)

    def toggle_play(self):
        self.paused = not self.paused
        self.play_button.setText("Pause" if not self.paused else "Play")

        if not self.paused:
            if self.tracking_point is None:
                QMessageBox.warning(self, "Warning", "Please set tracking point first using 'Set Tracking Point' button.")
                self.paused = True
                self.play_button.setText("Play")
                return

            self.timer.start(int(1000 / self.fps))
            self.status_bar.showMessage("Playing - tracking UAV...")
        else:
            self.timer.stop()
            self.status_bar.showMessage("Paused")

    def toggle_trajectory(self):
        self.show_trajectory = not self.show_trajectory
        self.traj_button.setText("Hide Trajectory" if self.show_trajectory else "Show Trajectory")
        if self.current_frame is not None:
            self.display_frame(self.current_frame)

    def set_scale(self):
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_img)

        scale_dialog = QtWidgets.QDialog(self)
        scale_dialog.setWindowTitle("Set Scale")
        label = QLabel(scale_dialog)
        label.setPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio))
        layout = QVBoxLayout()
        layout.addWidget(label)
        scale_dialog.setLayout(layout)

        self.scale_points = []

        def on_click(event):
            if len(self.scale_points) < 2:
                x = event.pos().x() * pixmap.width() / label.width()
                y = event.pos().y() * pixmap.height() / label.height()
                self.scale_points.append(QPoint(int(x), int(y)))

                painter = QtGui.QPainter(label.pixmap())
                painter.setPen(QtGui.QPen(Qt.red, 5))
                painter.drawPoint(int(x * label.width() / pixmap.width()),
                                  int(y * label.height() / pixmap.height()))
                painter.end()
                label.update()

                if len(self.scale_points) == 2:
                    painter = QtGui.QPainter(label.pixmap())
                    painter.setPen(QtGui.QPen(Qt.green, 2))
                    p1 = self.scale_points[0]
                    p2 = self.scale_points[1]
                    painter.drawLine(
                        int(p1.x() * label.width() / pixmap.width()),
                        int(p1.y() * label.height() / pixmap.height()),
                        int(p2.x() * label.width() / pixmap.width()),
                        int(p2.y() * label.height() / pixmap.height()))
                    painter.end()
                    label.update()

                    length, ok = QInputDialog.getDouble(
                        scale_dialog, "Enter Length",
                        "Enter the real-world length in meters:",
                        1.0, 0.01, 100.0, 2)
                    if ok:
                        dx = p2.x() - p1.x()
                        dy = p2.y() - p1.y()
                        pixel_length = (dx**2 + dy**2)**0.5
                        self.scale_factor = length / pixel_length
                        self.status_bar.showMessage(f"Scale set: 1 pixel = {self.scale_factor:.6f} meters")

                    scale_dialog.accept()

        label.mousePressEvent = on_click
        scale_dialog.exec_()

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

    def update_table(self):
        if len(self.positions) < 1:
            return

        self.table.setRowCount(len(self.positions))

        for i in range(len(self.positions)):
            dist = sum(
                ((self.positions[j][0] - self.positions[j-1][0])**2 +
                 (self.positions[j][1] - self.positions[j-1][1])**2)**0.5 * self.scale_factor
                for j in range(1, i+1)) if i > 0 else 0

            time_val = i / self.user_declared_fps

            vel = 0.0
            if i > 0 and i-1 < len(self.velocities):
                vel = self.velocities[i-1]

            accel = 0.0
            if i > 1 and i-2 < len(self.accelerations):
                accel = self.accelerations[i-2]

            self.table.setItem(i, 0, QTableWidgetItem(f"{dist:.3f}"))
            self.table.setItem(i, 1, QTableWidgetItem(f"{time_val:.3f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{vel:.3f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{accel:.3f}"))

    def export_data(self):
        if len(self.positions) < 1:
            QtWidgets.QMessageBox.warning(self, "Error", "Not enough data to export.")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Data", "",
            "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)",
            options=options)

        if file_path:
            try:
                data = {
                    "Frame": [],
                    "X (px)": [],
                    "Y (px)": [],
                    "Distance (m)": [],
                    "Time (s)": [],
                    "Velocity (m/s)": [],
                    "Acceleration (m/s²)": []
                }

                for i in range(len(self.positions)):
                    dist = 0.0
                    if i > 0:
                        prev_x, prev_y = self.positions[i-1]
                        curr_x, curr_y = self.positions[i]
                        step_dist = np.hypot(curr_x - prev_x, curr_y - prev_y) * self.scale_factor
                        dist = data["Distance (m)"][i-1] + step_dist

                    data["Frame"].append(i+1)
                    data["X (px)"].append(self.positions[i][0])
                    data["Y (px)"].append(self.positions[i][1])
                    data["Distance (m)"].append(dist)
                    data["Time (s)"].append(self.timestamps[i])
                    data["Velocity (m/s)"].append(self.velocities[i] if i < len(self.velocities) else 0.0)
                    data["Acceleration (m/s²)"].append(self.accelerations[i] if i < len(self.accelerations) else 0.0)

                df = pd.DataFrame(data)

                metadata_dict = {
                    "Video Path": self.video_path,
                    "Video Resolution": f"{int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
                    "Analysis FPS": self.user_declared_fps,
                    "Scale Factor (m/px)": f"{self.scale_factor:.6f}",
                    "Export Date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                if file_path.endswith('.xlsx'):
                    with pd.ExcelWriter(file_path) as writer:
                        pd.DataFrame(list(metadata_dict.items()), columns=['Parameter', 'Value']).to_excel(
                            writer,
                            sheet_name='Metadata',
                            index=False
                        )

                        df.to_excel(
                            writer,
                            sheet_name='Tracking Data',
                            index=False,
                            float_format="%.3f"
                        )

                else:  # CSV
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write("# UAV Tracking Analysis Export\n")
                        for key, value in metadata_dict.items():
                            f.write(f"# {key}: {value}\n")
                        f.write("#\n")

                        df.to_csv(
                            f,
                            index=False,
                            float_format="%.3f"
                        )

                QtWidgets.QMessageBox.information(self, "Success", f"Data exported successfully to:\n{file_path}")

            except Exception as e:
                error_msg = f"Export error: {str(e)}\n\n{traceback.format_exc()}"
                QtWidgets.QMessageBox.critical(self, "Error", error_msg)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = UAVAnalyzer()
    window.show()
    sys.exit(app.exec_())