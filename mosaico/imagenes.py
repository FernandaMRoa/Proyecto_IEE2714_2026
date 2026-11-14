"Lectura y escritura de imagenes RGB de 8 bits."

import numpy as np
from skimage import io


def a_rgb8(img):
    """Transforma la imagen a RGB uint8 (H, W, 3).
    gris -> se replica en 3 canales
    RGBA -> se descarta alfa. 16 bits o float -> se reescala a 0..255."""

    img = np.asarray(img)

    # Todo el proyecto asume 3 canales
    if img.ndim == 2:  # gris
        img = np.stack([img] * 3, axis=2)

    if img.shape[2] == 4:  # RGBA
        img = img[:, :, :3]

    if img.shape[2] == 2:  # gris + alfa
        img = np.stack([img[:, :, 0]] * 3, axis=2)

    # Tipo y escala: todo debe quedar en 0..255 y el negro debe seguir siendo 0
    if img.dtype == np.uint8:
        return np.ascontiguousarray(img)  # el recorte [:, :, :3] deja una vista no continua

    if img.dtype == np.uint16:  # 65535 = 255 * 257: el maximo es justo 255 y 0 sigue en 0
        return (img // 257).astype(np.uint8)

    if np.issubdtype(img.dtype, np.floating):
        m = 1.0 if img.max() <= 1.0 else 255.0  # rango [0, 1] o [0, 255]
        # round en vez de truncar. clip para que valores fuera de rango no den la vuelta
        return np.clip(np.round(img / m * 255), 0, 255).astype(np.uint8)

    # otros enteros (int32, int64): recortar al rango antes de convertir
    return np.clip(img, 0, 255).astype(np.uint8)


def leer_rgb(ruta):
    "Lee una imagen de la ruta como RGB uint8"
    return a_rgb8(io.imread(ruta))


def guardar_rgb(ruta, img):
    """Guarda la imagen sin perdidas. Usar .png o .tif: con JPEG los ceros de las
    franjas dejan de ser exactamente cero y la estimacion del interleave se complica."""
    io.imsave(ruta, np.asarray(img, dtype=np.uint8), check_contrast=False)
