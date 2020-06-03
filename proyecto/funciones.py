import numpy as np
from scipy import fftpack
from scipy.signal import medfilt

def elimina_ruido_impulsivo(freq, frame): 
    frame = medfilt(frame, 5)
    
    return frame

def elimina_ruido_periodico(freq, frame): 
    frame_freq = fftpack.fftshift(fftpack.fft2(frame))
    frame_freq[230-freq:260-freq, 418:427] = 0
    frame_freq[230+freq:260+freq, 418:427] = 0
    frame_limpio = np.abs(fftpack.ifft2(fftpack.ifftshift(frame_freq)))
        
    return frame_limpio


def eliminar_ruido(frame):

    freq = 28
    frame = elimina_ruido_impulsivo(freq, frame)
    frame_limpio = elimina_ruido_periodico(freq, frame)
    
    return frame_limpio

def transmisor(frame):
    # Implemente los bloques de
    #
    # Transformación
    #
    # Cuantización
    #
    # Codificación de fuente
    #
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