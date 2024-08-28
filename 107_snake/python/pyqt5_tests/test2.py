import sys
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsRectItem, QApplication
from PyQt5.QtGui import QBrush, QPen
from PyQt5.QtCore import Qt

app = QApplication(sys.argv)

# Defining a scene rect of 400x200, with it's origin at 0,0.
# If we don't set this on creation, we can set it later with .setSceneRect
scene = QGraphicsScene(0, 0, 400, 400)

# Draw a rectangle item, setting the dimensions.
rect = QGraphicsRectItem(0, 0, 380, 380)

# Set the origin (position) of the rectangle in the scene.
rect.setPos(0, 0)

# Define the brush (fill).
brush = QBrush(Qt.red)
rect.setBrush(brush)

# Define the pen (line)
#pen = QPen(Qt.cyan)
#pen.setWidth(10)
#rect.setPen(pen)

scene.addItem(rect)

view = QGraphicsView(scene)
view.show()
app.exec_()

print("done")