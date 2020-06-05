import numpy as np
from scipy import fftpack
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import cv2
from collections import Counter
import heapq

def elimina_ruido_impulsivo(frame): 
    return medfilt(frame, 5)
    
def elimina_ruido_periodico(freq, frame): 
    frame_freq = fftpack.fftshift(fftpack.fft2(frame))
    frame_freq[230-freq:260-freq, 418:427] = 0
    frame_freq[230+freq:260+freq, 418:427] = 0
    
    return np.abs(fftpack.ifft2(fftpack.ifftshift(frame_freq)))
        
    
def eliminar_ruido(frame):
    freq = 25
    frame = elimina_ruido_impulsivo(frame)
    frame_limpio = elimina_ruido_periodico(freq, frame)
    
    return frame_limpio
  

def transforma_frame(frame): 
    DCT2 = lambda g, norm='ortho': fftpack.dct( fftpack.dct(g, axis=0, norm=norm), axis=1, norm=norm)
    frame_size = frame.shape
    dct_matrix = np.zeros(shape = frame_size)

    for i in range(0, frame_size[0], 8):
        for j in range(0, frame_size[1], 8):
            dct_matrix[i:(i+8),j:(j+8)] = DCT2(frame[i:(i+8),j:(j+8)])
    
    return dct_matrix

def cuantiza_frame(frame, frame_min, q):
    return frame_min + q/2 + np.floor((frame - frame_min) / q) * q

def count(frame):
    f = np.ravel(frame.astype(np.uint8))
    return Counter(f).most_common()

def huffman(frame):
    dendograma = [[frequencia/len(frame), [simbolo, ""]] for simbolo, frequencia in count(frame)]
    heapq.heapify(dendograma)
    while len(dendograma) > 1:
        lo = heapq.heappop(dendograma)
        hi = heapq.heappop(dendograma)
        for codigo in lo[1:]:
            codigo[1] = '0' + codigo[1]
        for codigo in hi[1:]:
            codigo[1] = '1' + codigo[1]
        heapq.heappush(dendograma, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    dendograma = sorted(heapq.heappop(dendograma)[1:])
    dendograma = {simbolo : codigo for simbolo, codigo in dendograma} 
    return dendograma
    #display(dendograma)

    
def transmisor(frame):
    frame_transformado = transforma_frame(frame)
    frame_cuantizado = cuantiza_frame(frame_transformado, frame_transformado.min(), q = 0.01)
    
    h=huffman(frame_cuantizado)
    frame_comprimido = frame
    
    return frame_comprimido


def receptor(frame_comprimido):
    # Implemente los bloques de
    #
    # Decodificación
    #
    # Transformación inversa
    #
    frame = frame_comprimido
    return frame