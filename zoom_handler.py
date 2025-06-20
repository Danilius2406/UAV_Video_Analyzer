from PyQt5.QtCore import Qt, QPoint, pyqtSignal
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPainter, QPixmap, QWheelEvent

class ZoomableVideoWidget(QLabel):
    zoomChanged = pyqtSignal()
    clicked = pyqtSignal(QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.zoom_factor = 1.0
        self.base_pixmap = None
        self.offset = QPoint(0, 0)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.dragging = False
        self.drag_start_pos = QPoint()
        self.allow_pan = True

    def setPixmap(self, pixmap):
        try:
            self.base_pixmap = pixmap.copy() if pixmap else None
            self.update_display()
        except Exception as e:
            print(f"Pixmap error: {str(e)}")

    def wheelEvent(self, event: QWheelEvent):
        zoom_speed = 1.10
        mouse_pos = event.pos()

        old_pos = self.mapToSource(mouse_pos)

        if event.angleDelta().y() > 0:
            self.zoom_factor *= zoom_speed
        else:
            self.zoom_factor = max(self.zoom_factor / zoom_speed, 0.5)

        new_pos = self.mapFromSource(old_pos)
        self.offset += mouse_pos - new_pos

        self.update_display()
        self.zoomChanged.emit()
        event.accept()

    def mousePressEvent(self, event):
        if self.allow_pan and event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_start_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        else:
            source_pos = self.mapToSource(event.pos())
            self.clicked.emit(source_pos)

    def mouseMoveEvent(self, event):
        if self.dragging and self.allow_pan:
            delta = event.pos() - self.drag_start_pos
            self.offset += delta
            self.drag_start_pos = event.pos()
            self.update_display()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def mapToSource(self, pos):
        if not self.base_pixmap:
            return QPoint(0, 0)

        return QPoint(
            int((pos.x() - self.offset.x()) / self.zoom_factor),
            int((pos.y() - self.offset.y()) / self.zoom_factor)
        )

    def mapFromSource(self, pos):
        return QPoint(
            int(pos.x() * self.zoom_factor + self.offset.x()),
            int(pos.y() * self.zoom_factor + self.offset.y())
        )

    def update_display(self):
        if not self.base_pixmap:
            return

        scaled_pixmap = self.base_pixmap.scaled(
            self.base_pixmap.size() * self.zoom_factor,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        final_pixmap = QPixmap(self.size())
        final_pixmap.fill(Qt.white)

        painter = QPainter(final_pixmap)
        painter.drawPixmap(
            self.offset.x(),
            self.offset.y(),
            scaled_pixmap
        )
        painter.end()

        super().setPixmap(final_pixmap)