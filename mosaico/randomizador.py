"""Randomizador de parches: genera casos de desarrollo con su ground truth (seccion 2).

Pasos: crop centrado de R*s_h x C*s_w -> R*C celdas -> cada parche p_h x p_w se copia a su celda
con q ceros antes (separacion) -> permutacion aleatoria de las celdas con semilla fija.

Convencion del ground truth:
    P_GT es una matriz R x C de enteros 1-based.
    P_GT[r, c] = k de la pieza de la grilla de ENTRADA que va en la posicion (r, c) de la
    imagen correcta (SALIDA), con k = (r_s - 1) * C + c_s.

El ground truth se guarda en un archivo separado (gt/gt.json) que el ensamblador nunca lee,
solo lo usa evaluar.py despues del ensamblado.
"""

import numpy as np
from .geometria import geometria


def crop_centrado(img, H, W):
    "Recorte centrado de tamaño H x W"
    H0, W0 = img.shape[:2]
    f0, c0 = (H0 - H) // 2, (W0 - W) // 2
    return img[f0:f0 + H, c0:c0 + W], (f0, c0)


def cortar_parches(img, geo):
    """Divide una imagen ya recortada (H x W) en R*C parches utiles p_h x p_w.
    El parche de la celda (i, j) empieza en (i*s_h + q, j*s_w + q). Los pixeles que caen en las
    franjas de separacion se descartan."""

    R, C, q = geo['R'], geo['C'], geo['q']
    s_h, s_w, p_h, p_w = geo['s_h'], geo['s_w'], geo['p_h'], geo['p_w']
    parches = np.empty((R * C, p_h, p_w, img.shape[2]), dtype=img.dtype)

    for i in range(R):
        for j in range(C):
            y, x = i * s_h + q, j * s_w + q
            parches[i * C + j] = img[y:y + p_h, x:x + p_w]
    return parches


def armar_mosaico(parches, geo, orden=None):
    """Pone los parches en un lienzo H x W de ceros.

    orden[pos] indica el indice (desde 0) del parche que se coloca en la
    celda pos, recorriendo las celdas por filas. Si no se especifica orden,
    los parches se colocan en su orden original.
    """

    R, C, q = geo['R'], geo['C'], geo['q']
    s_h, s_w, p_h, p_w = geo['s_h'], geo['s_w'], geo['p_h'], geo['p_w']
    lienzo = np.zeros((geo['H'], geo['W'], parches.shape[3]), dtype=parches.dtype)

    if orden is None:
        orden = np.arange(R * C)

    orden = np.asarray(orden).ravel()
    for pos in range(R * C):
        i, j = divmod(pos, C)
        y, x = i * s_h + q, j * s_w + q
        lienzo[y:y + p_h, x:x + p_w] = parches[orden[pos]]
    return lienzo


def generar_caso(img, R, C, d, semilla, estricto=True):
    """Genera un caso: mosaico desordenado + ground truth + un anclaje aleatorio.

    Retorna (mosaico, publico, gt):
      mosaico : RGB uint8 H x W (entrada del ensamblador)
      publico : dict con lo unico que puede ver el ensamblador (R, C y anclaje 1-based)
      gt      : dict con la permutacion verdadera P_GT, d, semilla, geometria y crop
    """
    H0, W0 = img.shape[:2]
    geo = geometria(H0, W0, R, C, d, estricto=estricto) # entrega ValueError si no es valida
    crop, (f0, c0) = crop_centrado(img, geo['H'], geo['W'])
    parches = cortar_parches(crop, geo) # parches en su posicion verdadera

    rng = np.random.default_rng(semilla)
    perm = rng.permutation(R * C) # perm[s] = posicion verdadera del parche puesto en s
    mosaico = armar_mosaico(parches, geo, orden=perm)

    # P_GT[t] = s (posicion de entrada que contiene la pieza verdadera t), en 1-based
    P_gt = np.empty(R * C, dtype=int)
    P_gt[perm] = np.arange(R * C)
    P_gt = (P_gt + 1).reshape(R, C)

    # anclaje arbitrario: cualquier pieza
    s = int(rng.integers(R * C))
    t = int(perm[s])
    anclaje = [s // C + 1, s % C + 1, t // C + 1, t % C + 1] # (rs, cs, rt, ct) 1-based

    publico = {'R': R, 'C': C, 'anclaje': anclaje}
    gt = {'P_gt': P_gt.tolist(), 'd': d, 'semilla': int(semilla), 'geometria': geo,
          'crop_origen': [int(f0), int(c0)], 'tamano_original': [int(H0), int(W0)]}
    return mosaico, publico, gt


def referencia(img, R, C, d):
    """Imagen verdadera de comparacion: el crop con las franjas de separacion en cero
    (es lo que deberia devolver el ensamblador si acierta todo)."""
    
    geo = geometria(img.shape[0], img.shape[1], R, C, d, estricto=False)
    crop, _ = crop_centrado(img, geo['H'], geo['W'])
    return armar_mosaico(cortar_parches(crop, geo), geo)
