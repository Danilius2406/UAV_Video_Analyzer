import cv2
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (QFileDialog, QVBoxLayout, QPushButton,
                             QLabel, QHBoxLayout, QTableWidget,
                             QTableWidgetItem, QMessageBox, QInputDialog,
                             QSpinBox, QFormLayout)
from PyQt5.QtCore import QTimer, Qt, QPoint
import sys
import time
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
        self.tracking_radius = 15
        self.search_radius = 100  # Increased search radius for better tracking
        self.template = None
        self.template_size = 50  # Larger template for better matching
        self.user_declared_fps = 30.0
        self.current_frame = None
        self.tracking_active = False
        self.prev_gray = None
        self.setFocusPolicy(Qt.StrongFocus)
        self.trajectory = []  # Stores all historical points
        self.show_trajectory = False
        self.trajectory_color = (0, 0, 255)  # red path
        self.trajectory_thickness = 2
        self.dash_length = 10  # Length of each dash segment
        self.gap_length = 5    # Length of gap between dashes
        self.max_trajectory_points = 1000  # Maximum points to keep in trajectory

    def initUI(self):
        self.setWindowTitle("UAV Tracking Analyzer")
        self.setGeometry(100, 100, 1200, 800)

        # Main widgets
        self.video_label = ZoomableVideoWidget(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.mousePressEvent = self.video_label_clicked

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

        self.traj_style_button = QPushButton("Trajectory Style", self)
        self.traj_style_button.clicked.connect(self.set_trajectory_style)

        self.clear_traj_button = QPushButton("Clear Trajectory", self)
        self.clear_traj_button.clicked.connect(self.clear_trajectory)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.prev_frame_button)
        button_layout.addWidget(self.next_frame_button)
        button_layout.addWidget(self.set_scale_button)
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

        # Status bar
        self.status_bar = self.statusBar()

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
        """Draw a dashed line between two points"""
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

    def video_label_clicked(self, event):
        try:
            if self.current_frame is None or not self.video_label.base_pixmap:
                return

            source_pos = self.video_label.mapToSource(event.pos())
            x = source_pos.x()
            y = source_pos.y()

            frame_height, frame_width = self.current_frame.shape[:2]
            x = max(0, min(frame_width-1, x))
            y = max(0, min(frame_height-1, y))

            self.tracking_point = QPoint(x, y)

            frame_gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
            half_size = self.template_size // 2

            x1 = max(0, x - half_size)
            y1 = max(0, y - half_size)
            x2 = min(frame_width, x + half_size)
            y2 = min(frame_height, y + half_size)

            self.template = frame_gray[y1:y2, x1:x2]
            self.tracking_active = True

            marker_frame = self.current_frame.copy()
            cv2.drawMarker(marker_frame, (x, y), (0, 255, 255),
                           markerType=cv2.MARKER_CROSS,
                           markerSize=20,
                           thickness=2)
            self.display_frame(marker_frame)
            self.status_bar.showMessage(f"Tracking point: ({x}, {y})")

        except Exception as e:
            self.status_bar.showMessage(f"Error: {str(e)}")

    def load_video(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)",
            options=options)

        if file_path:
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

            # Draw tracking point (red circle)
            cv2.circle(frame_rgb, (x, y), self.tracking_radius, (255, 0, 0), 2)

            # Draw template area (green rectangle)
            cv2.rectangle(frame_rgb,
                          (x - self.template_size//2, y - self.template_size//2),
                          (x + self.template_size//2, y + self.template_size//2),
                          (0, 255, 0), 1)

        # Display frame
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line,
                             QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap)

    def update_frame(self):
        if self.paused or not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            self.export_button.setEnabled(True)
            self.status_bar.showMessage("Video processing complete. Ready to export data.")
            return

        self.current_frame = frame.copy()
        self.frame_count += 1
        current_time = self.frame_count / self.user_declared_fps

        if self.tracking_active and self.tracking_point and self.template is not None:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Add current point to trajectory (limit size)
            self.trajectory.append((self.tracking_point.x(), self.tracking_point.y()))
            if len(self.trajectory) > self.max_trajectory_points:
                self.trajectory.pop(0)

            # Get current tracking position
            x, y = self.tracking_point.x(), self.tracking_point.y()

            # Define search area
            margin = self.search_radius
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame_gray.shape[1], x + margin)
            y2 = min(frame_gray.shape[0], y + margin)

            search_area = frame_gray[y1:y2, x1:x2]

            # Perform template matching
            try:
                res = cv2.matchTemplate(search_area, self.template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                # Update tracking point
                new_x = x1 + max_loc[0] + self.template.shape[1] // 2
                new_y = y1 + max_loc[1] + self.template.shape[0] // 2
                self.tracking_point = QPoint(new_x, new_y)

                # Store position and time
                if len(self.positions) < self.frame_count:
                    self.positions.append((new_x, new_y))
                    self.timestamps.append(current_time)
                else:
                    self.positions[self.frame_count-1] = (new_x, new_y)
                    self.timestamps[self.frame_count-1] = current_time

                # Calculate metrics
                if len(self.positions) > 1:
                    prev_x, prev_y = self.positions[-2]
                    curr_x, curr_y = self.positions[-1]
                    distance_px = ((curr_x - prev_x)**2 + (curr_y - prev_y)**2)**0.5
                    distance_m = distance_px * self.scale_factor

                    time_diff = current_time - self.timestamps[-2]
                    if time_diff > 0:
                        velocity = distance_m / time_diff
                        if len(self.velocities) < len(self.positions) - 1:
                            self.velocities.append(velocity)
                        else:
                            self.velocities[-1] = velocity

                        # Calculate acceleration
                        if len(self.velocities) > 1:
                            velocity_diff = self.velocities[-1] - self.velocities[-2]
                            acceleration = velocity_diff / time_diff
                            if len(self.accelerations) < len(self.velocities) - 1:
                                self.accelerations.append(acceleration)
                            else:
                                self.accelerations[-1] = acceleration

                # Update template every 5 frames
                if self.frame_count % 5 == 0:
                    half_size = self.template_size // 2
                    x1 = max(0, new_x - half_size)
                    y1 = max(0, new_y - half_size)
                    x2 = min(frame_gray.shape[1], new_x + half_size)
                    y2 = min(frame_gray.shape[0], new_y + half_size)
                    self.template = frame_gray[y1:y2, x1:x2]

                self.prev_gray = frame_gray

            except Exception as e:
                self.status_bar.showMessage(f"Tracking error: {str(e)}")

        self.display_frame(frame)
        self.update_table()

    def prev_frame(self):
        if self.cap is not None and self.frame_count > 0:
            self.paused = True
            self.play_button.setText("Play")
            self.frame_count -= 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()

                # Update tracking point if we have data for this frame
                if len(self.positions) > self.frame_count:
                    x, y = self.positions[self.frame_count]
                    self.tracking_point = QPoint(x, y)

                    # Update template for better tracking when resuming
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    half_size = self.template_size // 2
                    x1 = max(0, x - half_size)
                    y1 = max(0, y - half_size)
                    x2 = min(frame_gray.shape[1], x + half_size)
                    y2 = min(frame_gray.shape[0], y + half_size)
                    self.template = frame_gray[y1:y2, x1:x2]

                    self.tracking_active = True  # Keep tracking activ

                self.display_frame(frame)
                self.status_bar.showMessage(f"Frame {self.frame_count+1}: Position ({x}, {y})")

    def next_frame(self):
        if self.cap is not None:
            self.paused = True
            self.play_button.setText("Play")
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                self.frame_count += 1

                # Update tracking point if we have data for this frame
                if len(self.positions) > self.frame_count - 1:
                    x, y = self.positions[self.frame_count - 1]
                    self.tracking_point = QPoint(x, y)

                    # Update template for better tracking when resuming
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    half_size = self.template_size // 2
                    x1 = max(0, x - half_size)
                    y1 = max(0, y - half_size)
                    x2 = min(frame_gray.shape[1], x + half_size)
                    y2 = min(frame_gray.shape[0], y + half_size)
                    self.template = frame_gray[y1:y2, x1:x2]

                    self.tracking_active = True  # Keep tracking active

                self.display_frame(frame)
                self.status_bar.showMessage(f"Frame {self.frame_count+1}: Position ({x}, {y})")

    def toggle_play(self):
        self.paused = not self.paused
        self.play_button.setText("Pause" if not self.paused else "Play")

        if not self.paused:
            if self.tracking_point is None:
                QMessageBox.warning(self, "Warning", "Please click on the UAV to set tracking point first.")
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

        # Store current position
        current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Convert frame for display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_img)

        # Create scale dialog
        scale_dialog = QtWidgets.QDialog(self)
        scale_dialog.setWindowTitle("Set Scale")
        label = QLabel(scale_dialog)
        label.setPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio))
        layout = QVBoxLayout()
        layout.addWidget(label)
        scale_dialog.setLayout(layout)

        # Get two points from user
        self.scale_points = []

        def on_click(event):
            if len(self.scale_points) < 2:
                x = event.pos().x() * pixmap.width() / label.width()
                y = event.pos().y() * pixmap.height() / label.height()
                self.scale_points.append(QPoint(int(x), int(y)))

                # Draw point
                painter = QtGui.QPainter(label.pixmap())
                painter.setPen(QtGui.QPen(Qt.red, 5))
                painter.drawPoint(int(x * label.width() / pixmap.width()),
                                  int(y * label.height() / pixmap.height()))
                painter.end()
                label.update()

                if len(self.scale_points) == 2:
                    # Draw line
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

                    # Ask for real length
                    length, ok = QInputDialog.getDouble(
                        scale_dialog, "Enter Length",
                        "Enter the real-world length in meters:",
                        1.0, 0.01, 100.0, 2)
                    if ok:
                        # Calculate scale factor
                        dx = p2.x() - p1.x()
                        dy = p2.y() - p1.y()
                        pixel_length = (dx**2 + dy**2)**0.5
                        self.scale_factor = length / pixel_length
                        self.status_bar.showMessage(f"Scale set: 1 pixel = {self.scale_factor:.6f} meters")

                    scale_dialog.accept()

        label.mousePressEvent = on_click
        scale_dialog.exec_()

        # Return to original frame position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

    def update_table(self):
        if len(self.positions) < 1:
            return

        self.table.setRowCount(len(self.positions))

        for i in range(len(self.positions)):
            # Calculate cumulative distance
            dist = sum(
                ((self.positions[j][0] - self.positions[j-1][0])**2 +
                 (self.positions[j][1] - self.positions[j-1][1])**2)**0.5 * self.scale_factor
                for j in range(1, i+1)) if i > 0 else 0

            time_val = i / self.user_declared_fps

            vel = 0.0
            if i > 0 and i-1 < len(self.velocities):
                vel = self.velocities[i-1]

            # Acceleration calculation
            accel = 0.0
            if i > 1 and i-2 < len(self.accelerations):
                accel = self.accelerations[i-2]

            self.table.setItem(i, 0, QTableWidgetItem(f"{dist:.3f}"))
            self.table.setItem(i, 1, QTableWidgetItem(f"{time_val:.3f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{vel:.3f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{accel:.3f}"))

    def export_data(self):
        if len(self.positions) < 1:
            QMessageBox.warning(self, "Error", "Not enough data to export.")
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
                    x, y = self.positions[i]
                    dist = sum(
                        ((self.positions[j][0] - self.positions[j-1][0])**2 +
                         (self.positions[j][1] - self.positions[j-1][1])**2)**0.5 * self.scale_factor
                        for j in range(1, i+1)) if i > 0 else 0

                    time_val = self.timestamps[i]
                    vel = self.velocities[i-1] if i > 0 and i-1 < len(self.velocities) else 0
                    accel = self.accelerations[i-2] if i > 1 and i-2 < len(self.accelerations) else 0

                    data["Frame"].append(i+1)
                    data["X (px)"].append(x)
                    data["Y (px)"].append(y)
                    data["Distance (m)"].append(dist)
                    data["Time (s)"].append(time_val)
                    data["Velocity (m/s)"].append(vel)
                    data["Acceleration (m/s²)"].append(accel)

                df = pd.DataFrame(data)

                if file_path.endswith('.xlsx'):
                    df.to_excel(file_path, index=False)
                else:
                    df.to_csv(file_path, index=False)

                QMessageBox.information(self, "Success", f"Data exported to {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export data: {str(e)}")

    def closeEvent(self, event):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = UAVAnalyzer()
    window.show()
    sys.exit(app.exec_())