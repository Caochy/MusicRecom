from ui.SongWindow import SongWindow
from util.UrlToPixmap import url_to_pixmap
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtGui import QIcon

class SongButton(QPushButton):
    def __init__(self, parent, song_info):
        super().__init__(parent)
        self.song_info = song_info
        str = self.song_info["song_title"] + " - " + self.song_info["singer_name"]
        super().setText(str)
        self.setFixedSize(300, 50)
        pass

    def mousePressEvent(self, e):
        self.sw = SongWindow(self.song_info)
        print(1)
        self.sw.show()
        print(2)
        pass

    def update_song(self, s):
        self.song_info = s
        str = self.song_info["song_title"] + " - " + self.song_info["singer_name"]
        super().setText(str)
