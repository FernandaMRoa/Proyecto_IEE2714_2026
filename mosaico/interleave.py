"""Estimacion automatica del interleave d y extraccion de los parches (secciones 1 y 3).

Como R y C son conocidos, el tamaño de celda s_h = H/R, s_w = W/C tambien lo es. Cada
candidato d predice exactamente que filas y columnas locales de cada celda son separacion
(q al inicio y q+1 al final). Para aceptar un candidato se exige que esas filas y columnas
sean cero en todas las celdas de la grilla, en ambas direcciones a la vez.

Una linea, region o bloque negro del contenido original de la imagen queda, despues de permutar, 
solo en algunas celdas y en posiciones locales arbitrarias. Para que el estimador sea engañado, 
todas las R*C piezas tendrian que tener contenido exactamente cero en las mismas filas y columnas 
locales (las que separan un candidato del siguiente). En vez de buscar filas de ceros aisladas en 
la imagen, se prueba la periodisidad de la estructura completa.

Los candidatos estan anidados, es decir, las separaciones de d=1 estan contenidas en las de d=3,
y estas en las de d=5, asi que se elige el mayor d consistente.
"""

import warnings
import numpy as np
from .geometria import INTERLEAVES_VALIDOS, filas_separacion_locales, geometria
from .randomizador import cortar_parches


def _fracciones_cero(img, R, C):
    """Fila local = numero de fila dentro de una celda (0..s_h-1), igual para todas las celdas.

    Para cada fila local u devuelve f(u): la fraccion de las R*C celdas en que esa fila esta
    completamente en cero (los 3 canales y todo el ancho de la celda). Idem g(v) para cada
    columna local v (0..s_w-1). Las filas/columnas de separacion valen 1 en todas las celdas."""

    H, W = img.shape[:2]
    s_h, s_w = H // R, W // C
    no_cero = img[:R * s_h, :C * s_w].max(axis=2) > 0
    bloques = no_cero.reshape(R, s_h, C, s_w)
    filas_cero = ~bloques.any(axis=3) # (R, s_h, C)
    cols_cero = ~bloques.any(axis=1) # (R, C, s_w)

    return filas_cero.mean(axis=(0, 2)), cols_cero.mean(axis=(0, 1))


def _perfiles_medios(img, R, C):
    """Perfil de brillo medio por fila local y por columna local (se usa solo en el modo tolerante).
    Para cada pixel se toma el maximo entre sus 3 canales como medida de brillo. Luego, para cada
    fila local u (0..s_h-1), se promedia ese brillo en toda la fila y en las R*C celdas.
    idem para cada columna local v (0..s_w-1).
    Sirve cuando las franjas no quedaron exactamente en cero, ya que las filas/columnas
    de separacion siguen siendo mucho mas oscuras que las de contenido."""

    H, W = img.shape[:2]
    s_h, s_w = H // R, W // C
    v = img[:R * s_h, :C * s_w].max(axis=2).astype(np.float64).reshape(R, s_h, C, s_w)
    return v.mean(axis=(0, 2, 3)), v.mean(axis=(0, 1, 2))


def estimar_interleave(img, R, C, candidatos=INTERLEAVES_VALIDOS, umbral_tolerante=20):
    """Estima el interleave d usando solo el mosaico.

    1. Para cada candidato d (0, 1, 3, 5) se sabe exactamente que filas y columnas locales de
    cada celda deberian ser separacion.

    2. Puntaje del candidato = fraccion minima de celdas en cero entre esas filas y columnas.
    Vale 1.0 solo si todas sus filas/columnas de separacion son cero en todas las celdas    
    (candidato consistente).

    3. Como los candidatos estan anidados, se elige el mayor d con puntaje 1.0.

    4. Si ningun d > 0 es consistente, se prueba el modo tolerante con el perfil de brillo medio.

    Devuelve (d, diagnostico). El diagnostico incluye:
      'modo'                     'estricto' o 'tolerante'
      'puntaje_por_candidato'    el puntaje de cada d. Ej. {'0': 1.0, '1': 1.0, '3': 1.0, '5': 0.0}
      'margen_siguiente'         puntaje del candidato inmediatamente mayor al elegido (None si se
                                 eligio d = 5). Cerca de 0: decision clara. Cerca de 1: ese
                                 candidato casi tambien cumplia, la decision es menos segura.
      'fraccion_contenido_negro' fraccion de filas/columnas de contenido completamente negras
                                 (indica que tan oscura es la imagen)
      's_h', 's_w'               tamano de las celdas"""
    
    H, W = img.shape[:2]
    if H % R or W % C:
        warnings.warn(f'La imagen {H}x{W} no es divisible por la grilla {R}x{C}; '
                      'se usa la region superior izquierda divisible.')
    s_h, s_w = H // R, W // C
    f_fila, f_col = _fracciones_cero(img, R, C)

    puntaje = {}
    for d in candidatos:
        if d == 0:
            puntaje[d] = 1.0
            continue
        if s_h - d < 2 or s_w - d < 2:
            puntaje[d] = 0.0
            continue
        filas = filas_separacion_locales(s_h, d)
        cols = filas_separacion_locales(s_w, d)
        puntaje[d] = float(min(f_fila[filas].min(), f_col[cols].min()))

    consistentes = [d for d in candidatos if puntaje[d] == 1.0]
    d_est = max(consistentes)
    modo = 'estricto'

    # Modo tolerante (solo si ningun d > 0 es exactamente consistente): por ejemplo si el
    # mosaico se guardo con compresion con perdidas (JPEG) y las franjas quedaron borrosas y no
    # exactamente en cero. Se usa el perfil de brillo medio por fila/columna local: para el d
    # correcto, las filas de separacion son claramente mas oscuras que todas las demas filas
    # (si d es muy pequeño, una fila oscura queda entre las "demas". Si d es muy grande, una fila de
    # contenido queda entre las "oscuras"). Se exige oscuro < 0.5 * claro en ambas direcciones.
    if d_est == 0:
        m_f, m_c = _perfiles_medios(img, R, C)
        mejor = None
        for d in candidatos:
            if d == 0 or s_h - d < 4 or s_w - d < 4:
                continue
            ff, cc = filas_separacion_locales(s_h, d), filas_separacion_locales(s_w, d)
            resto_f = np.delete(m_f, ff)
            resto_c = np.delete(m_c, cc)
            oscuro = max(m_f[ff].max(), m_c[cc].max())
            claro = min(resto_f.min(), resto_c.min())
            if claro >= umbral_tolerante and oscuro < 0.5 * claro:
                margen_t = (claro - oscuro) / claro
                if mejor is None or margen_t > mejor[0]:
                    mejor = (margen_t, d)
        if mejor is not None:
            d_est, modo = mejor[1], 'tolerante'

    mayores = [d for d in candidatos if d > d_est]
    margen = puntaje[min(mayores)] if mayores else None

    # fraccion de filas/columnas de contenido completamente negras (indica cuan oscura es la imagen)
    q5 = max(candidatos) // 2
    interior_f = f_fila[q5 + 1: s_h - q5 - 2]
    interior_c = f_col[q5 + 1: s_w - q5 - 2]
    fondo = float(np.mean(np.concatenate([interior_f, interior_c]))) if interior_f.size else 0.0

    diag = {'d': int(d_est), 'modo': modo, 'puntaje_por_candidato': {str(k): v for k, v in puntaje.items()},
            'margen_siguiente': margen, 'fraccion_contenido_negro': fondo, 's_h': int(s_h), 's_w': int(s_w)}
    return int(d_est), diag


def extraer_parches(img, R, C, d):
    """Extrae los R*C parches utiles (N, p_h, p_w, 3) en el orden de la grilla de entrada
    (fila mayor, indice 0-based k0 = r*C + c) y devuelve tambien la geometria."""
    H, W = img.shape[:2]
    geo = geometria(H - H % R, W - W % C, R, C, d, estricto=False)
    
    return cortar_parches(img[:geo['H'], :geo['W']], geo), geo
