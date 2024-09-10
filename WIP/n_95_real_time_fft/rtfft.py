import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from time import sleep, time
import scipy.signal as signal

fulldata = np.array([], dtype=np.int16)
fftdata = np.array([], dtype=np.int16)
time_0 = []
time_1 = []
count = 100

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
        global fulldata, fftdata, time_0, time_1, count
        audio_data = np.frombuffer(buffer=in_data, count=frame_count, dtype=np.int16)
        fulldata = np.append(fulldata, audio_data)
        time_0.append(time_info["input_buffer_adc_time"])
        time_1.append(time_info["current_time"])
        count -= 1
        return (audio_data, pyaudio.paContinue if count > 0 else pyaudio.paAbort)

    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        stream_callback=callback,
        frames_per_buffer=CHUNK,
    )
    sleep(2)
    while count > 0:
        sleep(0.01)
    t0 = time()
    time_array = np.linspace(min(time_0), max(time_1), fulldata.shape[0])

    print("stop here")

    fft = np.abs(np.fft.rfft(np.pad(fulldata, (1000, 1000), "constant")))
    N = len(fft)
    n = np.arange(N)
    T = N / RATE
    freq = n / T
    print(1000000 * (time() - t0))

    plt.plot(time_array, fulldata)
    plt.show()
    plt.plot(freq, fft)
    plt.show()
    print("stop")
