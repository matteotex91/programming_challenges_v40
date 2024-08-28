import numpy as np
from typing import Optional
import sys
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsRectItem, QApplication
from PyQt5.QtGui import QBrush, QPen
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeyEvent

""" This class contains all the informations and functions needed to plot and collect keyboard events
Snake directions :
0 -> right
1 -> up
2 -> left
3 -> down
"""
class SnakeGraphicsView(QGraphicsView):
    def __init__(self,map_shape:np.ndarray=np.array([25,25]),pixel_shape:np.ndarray=np.array([25,25]))->None:
        QGraphicsView.__init__(self)
        graphics_shape=map_shape*pixel_shape
        self._scene = QGraphicsScene(0, 0, graphics_shape[0],graphics_shape[1])
        self._snake_direction=0
        self._running=True
        self.show()


    def keyPressEvent(self,keyEvent:QKeyEvent)->None:
        #Process the event as this function was not overdriven
        super(SnakeGraphicsView,self).keyPressEvent(keyEvent)
        match keyEvent.key():
            case Qt.Key_D:
                self._snake_direction=0
            case Qt.Key_W:
                self._snake_direction=1
            case Qt.Key_A:
                self._snake_direction=2
            case Qt.Key_S:
                self._snake_direction=3
            case Qt.Key_Escape:
                self._running=False
                self.close()   
                   
    def redraw(map:np.ndarray,food_position:np.ndarray)->None:
        map_shape=map.shape()
        for ix in map_shape[0]:
            for iy in map_shape[1]:
                if 
        pass



if __name__=="__main__":
    app = QApplication(sys.argv)
    snake_graphics_view=SnakeGraphicsView()
    app.exec_()