from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtCore, QtGui
from ui.SongWindow import SongWindow

class SongLabel(QLabel):
    def __init__(self):
        super().__init__()
        pe = QPalette()
        pe.setColor(QPalette.WindowText, Qt.red)
        super().setPalette(pe)
        pass

    def set_song_info(self, si):
        self.song_info = si
        pass

    def mousePressEvent(self, ev):
        self.sw = SongWindow(self.song_info)
        self.layout().addWidget(self._widget)
        self.sw.show()
        pass