import matplotlib.pyplot as plt
import numpy as np
import pywt


def denoise(data):
    w = pywt.Wavelet('db8')
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    coeffs = pywt.wavedec(data, 'db8', level=maxlev)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], 0.3 * max(coeffs[i]))
    return pywt.waverec(coeffs, 'db8')


data = denoise(pywt.data.ecg())
plt.plot(np.arange(len(data)), data)
plt.xlabel('time (s)')
plt.ylabel('microvolts (uV)')
plt.title("De-noised signal using wavelet techniques")

plt.tight_layout()
plt.show()
