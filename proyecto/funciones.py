import numpy as np
from scipy import fftpack
from scipy.signal import medfilt

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