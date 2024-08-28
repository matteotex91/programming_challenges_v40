from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout,QMessageBox

def on_button_clicked():
    alert = QMessageBox()
    alert.setText('You clicked the button!')
    alert.exec()

app = QApplication([])
window = QWidget()
layout = QVBoxLayout()
button_top=QPushButton('Top')
button_bottom=QPushButton('Bottom')
layout.addWidget(button_top)
layout.addWidget(button_bottom)
window.setLayout(layout)

button_top.clicked.connect(on_button_clicked)

window.show()
app.exec()