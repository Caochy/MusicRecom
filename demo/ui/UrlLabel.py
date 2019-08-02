from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtCore, QtGui

class UrlLabel(QLabel):
    def __init__(self):
        super().__init__()
        pe = QPalette()
        pe.setColor(QPalette.WindowText, Qt.red)
        super().setPalette(pe)
        pass

    def set_url(self, u):
        self.url = u
        pass

    def mousePressEvent(self, ev):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(self.url))
        pass