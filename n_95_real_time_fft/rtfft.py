import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from time import sleep
import scipy.signal as signal

fulldata = np.array([], dtype=np.int16)
fftdata = np.array([], dtype=np.int16)
time_0 = []
time_1 = []

if __name__ == "__main__":
    # AUDIO INPUT
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 1
    WAVE_OUTPUT_FILENAME = "output.wav"

    audio = pyaudio.PyAudio()

    def callback(in_data, frame_count, time_info, flag):
        global fulldata, fftdata, time_0, time_1
        audio_data = np.frombuffer(buffer=in_data, count=frame_count, dtype=np.int16)
        fulldata = np.append(fulldata, audio_data)
        time_0.append(time_info["input_buffer_adc_time"])
        time_1.append(time_info["current_time"])
        return (audio_data, pyaudio.paContinue)

    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        stream_callback=callback,
        frames_per_buffer=CHUNK,
    )
    sleep(2)
    time_array = np.linspace(min(time_0), max(time_1), fulldata.shape[0])

    fft = np.fft.fft(a=fulldata, axis=time_array)
    plt.plot(time_array, fulldata)
    plt.show()
    plt.plot(fft)
    plt.show()
    print("stop")
