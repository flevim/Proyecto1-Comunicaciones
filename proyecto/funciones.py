import numpy as np
from scipy import fftpack
from scipy.signal import medfilt
def eliminar_ruido(frame,n,m):

    freq=28
    # Implemente esta función
    # ESTE CONDICION ES MOMENTANEA, ES PARA PROBAR CON FRAMES ESPECIFICOS
    if n==m:
        #Borra ruido impulsivo
        frame=medfilt(frame, 5)
        #Borra ruido periodico
        frame_freq = fftpack.fftshift(fftpack.fft2(frame))
        frame_freq[230-freq:260-freq,
                   418:427] = 0
        frame_freq[230+freq:260+freq,
                   418:427] = 0
        frame_limpio = np.abs(fftpack.ifft2(fftpack.ifftshift(frame_freq)))
    else:
        frame_limpio=frame

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