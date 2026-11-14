"""Visualizacion continua de la reconstruccion: relleno de las franjas de separacion (seccion 4).

Metodos:
  'lineal'   interpolacion lineal entre los bordes de los parches a ambos lados
             de cada franja, bilineal en los cruces (con las 4 esquinas de los parches
             diagonales) y extension del valor de borde en el marco externo. Se implementa como
             una interpolacion separable sobre la grilla de pixeles validos.

  'cubico'   interpolacion cubica de Hermite a traves de cada franja: ademas de los valores usa la
             derivada a cada lado (estimada con 2 pixeles), asi la transicion es C1 (continua en
             valor y pendiente) en vez de C0. Primero se rellenan las franjas verticales sobre las
             filas validas y luego las horizontales sobre todas las columnas (los cruces quedan
             interpolados en ambas direcciones).

  'armonico' solucion de la ecuacion de Laplace (Poisson con fuente cero) en los pixeles de las
             franjas, con condicion de Dirichlet en los pixeles de los parches: es la superficie
             mas suave que calza con todos los bordes a la vez. Se resuelve con un sistema
             disperso (Laplaciano de 5 puntos), restringido a las franjas.
El borde externo de la imagen usa extension del valor de borde (replicacion) en los tres metodos.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from .geometria import geometria


def mascara_valida_1d(n_celdas, s, p, q):
    "True en los indices (filas o columnas) que pertenecen a un parche"
    m = np.zeros(n_celdas * s, dtype=bool)
    for i in range(n_celdas):
        m[i * s + q: i * s + q + p] = True
    return m


def _vecinos_1d(valido):
    """Para cada indice: indice valido anterior (a), siguiente (b) y peso t en [0, 1] tal que el
    valor interpolado es (1-t) x[a] + t x[b]. En los indices validos a = b = i. Antes del primer
    parche y despues del ultimo se replica el borde (a = b = primer/ultimo valido)."""
    n = len(valido)
    idx = np.arange(n)
    pos = idx[valido]
    j = np.searchsorted(pos, idx, side='right') - 1 # posicion en 'pos' del ultimo valido <= i
    a = pos[np.clip(j, 0, len(pos) - 1)]
    b = pos[np.clip(j + 1, 0, len(pos) - 1)]
    b = np.where(valido, idx, b)
    t = np.where(b > a, (idx - a) / np.maximum(b - a, 1), 0.0)
    return a, b, t


def _lineal(img, vf, vc):
    "Interpolacion separable: lineal en franjas, bilineal en cruces, replicacion afuera"
    af, bf, tf = _vecinos_1d(vf)
    ac, bc, tc = _vecinos_1d(vc)
    x = img.astype(np.float64)
    tf, tc = tf[:, None, None], tc[None, :, None]
    return ((1 - tf) * (1 - tc) * x[af][:, ac] + (1 - tf) * tc * x[af][:, bc] +
            tf * (1 - tc) * x[bf][:, ac] + tf * tc * x[bf][:, bc])


def _hermite_eje(x, valido, eje):
    """Rellena a lo largo de 'eje' los indices no validos interiores con Hermite cubico, usando
    valor y derivada (diferencia de 2 pixeles) a cada lado. Los exteriores se replican."""
    x = np.moveaxis(x, eje, 0).copy()
    a, b, t = _vecinos_1d(valido)
    n = len(valido)
    for i in np.where(~valido)[0]:
        ia, ib, ti = a[i], b[i], t[i]
        if ia == ib:  # borde externo: replicar
            x[i] = x[ia]
            continue
        L = ib - ia
        da = x[ia] - x[ia - 1] if ia - 1 >= 0 and valido[ia - 1] else 0 * x[ia]
        db = x[ib + 1] - x[ib] if ib + 1 < n and valido[ib + 1] else 0 * x[ib]
        h00 = 2 * ti ** 3 - 3 * ti ** 2 + 1
        h10 = ti ** 3 - 2 * ti ** 2 + ti
        h01 = -2 * ti ** 3 + 3 * ti ** 2
        h11 = ti ** 3 - ti ** 2
        x[i] = h00 * x[ia] + h10 * L * da + h01 * x[ib] + h11 * L * db
    return np.moveaxis(x, 0, eje)


def _cubico(img, vf, vc):
    x = img.astype(np.float64)
    x[vf] = _hermite_eje(x[vf], vc, 1) # franjas verticales, solo en filas validas
    x = _hermite_eje(x, vf, 0) # franjas horizontales (incluye los cruces)
    return x


def _armonico(img, vf, vc):
    """Laplace en las franjas internas con Dirichlet en los parches. Se parte del relleno lineal
    (que da los valores del borde externo por replicacion) y se resuelve solo en las incognitas
    interiores: 4 u(p) - sum vecinos = 0."""
    x = _lineal(img, vf, vc)
    H, W = vf.size, vc.size
    valido = vf[:, None] & vc[None, :]
    f_in = np.where(vf)[0]
    c_in = np.where(vc)[0]
    interior = np.zeros((H, W), dtype=bool)
    interior[f_in[0]:f_in[-1] + 1, c_in[0]:c_in[-1] + 1] = True
    incog = interior & ~valido
    n = int(incog.sum())

    if n == 0:
        return x
    
    ind = -np.ones((H, W), dtype=np.int64)
    ind[incog] = np.arange(n)
    ys, xs = np.nonzero(incog)
    filas, cols, vals = [np.arange(n)], [np.arange(n)], [np.full(n, 4.0)]
    b = np.zeros((n, x.shape[2]))

    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        yy, xx = ys + dy, xs + dx # siempre dentro (incognitas interiores)
        k = ind[yy, xx]
        es_inc = k >= 0
        filas.append(np.arange(n)[es_inc])
        cols.append(k[es_inc])
        vals.append(-np.ones(es_inc.sum()))
        b[~es_inc] += x[yy[~es_inc], xx[~es_inc]] # vecino conocido -> lado derecho
    A = sparse.csc_matrix((np.concatenate(vals), (np.concatenate(filas), np.concatenate(cols))),
                          shape=(n, n))
    sol = spsolve(A, b)
    x[incog] = sol.reshape(n, -1)
    return x


def rellenar_separaciones(imagen, R, C, d, metodo='lineal'):
    """Devuelve una imagen uint8 del mismo tamaño que 'imagen' con las franjas de separacion
    rellenadas. 'imagen' es la reconstruccion (parches en su lugar y ceros en las franjas)."""
    H, W = imagen.shape[:2]
    if d == 0:
        return imagen.copy()
    geo = geometria(H, W, R, C, d, estricto=False)
    vf = mascara_valida_1d(R, geo['s_h'], geo['p_h'], geo['q'])
    vc = mascara_valida_1d(C, geo['s_w'], geo['p_w'], geo['q'])
    vf = np.concatenate([vf, np.zeros(H - vf.size, dtype=bool)])
    vc = np.concatenate([vc, np.zeros(W - vc.size, dtype=bool)])
    f = {'lineal': _lineal, 'cubico': _cubico, 'armonico': _armonico}[metodo]
    out = f(imagen, vf, vc)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
