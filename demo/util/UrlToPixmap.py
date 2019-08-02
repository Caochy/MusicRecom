from PyQt5.QtGui import QPixmap, QIcon
import urllib.request

def url_to_pixmap(url):
    data = urllib.request.urlopen(url).read()
    pixmap = QPixmap()
    pixmap.loadFromData(data)
    return pixmap

if __name__ == "__main__":
    icon = url_to_pixmap("http://y.gtimg.cn/music/photo_new/T002R120x120M000001qYTzY2oyDyZ.jpg")