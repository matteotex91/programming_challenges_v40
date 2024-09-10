import pyqtgraph as pg
import pyaudio
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import pyqtSignal, QEvent, QThread
from PyQt5.QtGui import QMouseEvent
import sys
import numpy as np
from time import sleep


class AudioListener(QThread):
    fft_signal = pyqtSignal(object)

    def __init__(self, main_window):
        QThread.__init__(self)
        self.fft_signal.connect(main_window.replot_fft)
        self.started.connect(main_window.start_acquisition)


class MainWindow(QMainWindow):
    count = 0

    def __init__(self):
        super().__init__()
        # Temperature vs time plot
        self.plot_graph = pg.PlotWidget()
        self.setCentralWidget(self.plot_graph)
        # self.fft_signal.connect(self.replot_fft)
        self.show()
        self.al = AudioListener(self)

    def start_acquisition(self):
        self.count = 100
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 1024
        audio = pyaudio.PyAudio()

        def callback(in_data, frame_count, time_info, flag):
            audio_data = np.frombuffer(
                buffer=in_data, count=frame_count, dtype=np.int16
            )
            fft = np.fft.rfft(audio_data)
            N = len(fft)
            n = np.arange(N)
            T = N / RATE
            freq = n / T
            min_freq_ind = np.argmin(np.abs(freq - 20))
            max_freq_ind = np.argmin(np.abs(freq - 20000))
            self.fft_signal.emit(
                [
                    freq[min_freq_ind:max_freq_ind],
                    np.abs(fft[min_freq_ind:max_freq_ind]),
                ]
            )
            self.count -= 1
            return (
                audio_data,
                pyaudio.paContinue if self.count > 0 else pyaudio.paAbort,
            )

        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            stream_callback=callback,
            frames_per_buffer=CHUNK,
        )
        while self.count > 0:
            sleep(0.1)

    def replot_fft(self, fft_signal):
        self.plot_graph.clear()
        self.plot_graph.plot(fft_signal[0], fft_signal[1])


app = QApplication(sys.argv)
main = MainWindow()
main.show()
app.exec()
