from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

filename = '/ghome/group07/C5-W6/First_Impressions_v3_multimodal/test/1/-9BZ8A9U7TE.000.wav'
samplerate, data = wavfile.read(filename)
print(f"data shape = {data.shape}")
length = data.shape[0] / samplerate
print(f"length = {length}s")

time = np.linspace(0., length, data.shape[0])
plt.plot(time, data)
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()
plt.savefig("try.png")