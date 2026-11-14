"""Descriptores de borde y medidas de (di)similitud entre parches (seccion 3).

Notacion: para una union horizontal i -> j (j a la derecha de i) se comparan:
    A1 = ultima columna de i,   A2 = penultima columna de i
    B1 = primera columna de j,  B2 = segunda columna de j
Las uniones verticales (j debajo de i) se reducen al mismo caso transponiendo los parches.

Entre A1 y B1 hay g = d + 1 pasos de pixel (d pixeles de separacion perdidos), por lo que toda
diferencia a traves de la union se divide por g para expresarla por pixel.

Medidas implementadas (todas devuelven D[i, j] >= 0, menor = mas compatible):
  'ssd'  : diferencia cuadratica media entre A1 y B1.
  'pred' : prediccion lineal a traves de la union (extrapolacion con la derivada del borde) y
           norma Lp con p < 1 (robusta a valores atipicos), simetrica.
  'mgc'  : compatibilidad Mahalanobis del gradiente. El gradiente normal al borde de i (A1 - A2)
           define una distribucion (media y covarianza 3x3 entre canales). El gradiente que
           cruza la union, (B1 - A1)/g, deberia ser una muestra de esa distribucion. Se suma la
           distancia de Mahalanobis evaluada desde ambos lados.
Todas se normalizan por el largo del borde para que uniones horizontales y verticales (parches
rectangulares) sean comparables.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage.color import rgb2lab


def transformar_color(parches, espacio='lab'):
    """parches uint8 (N, h, w, 3) -> float32 en el espacio pedido.
    'rgb': valores 0...255. 
    'lab': CIE L*a*b*. 
    'gris': luminancia."""

    x = parches.astype(np.float32)
    if espacio == 'rgb':
        return x
    
    if espacio == 'lab':
        N, h, w, _ = x.shape
        lab = rgb2lab(x.reshape(N * h, w, 3) / 255.0).astype(np.float32)
        return lab.reshape(N, h, w, 3)
    
    if espacio == 'gris':
        g = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
        return g[..., None]
    
    raise ValueError(f'espacio de color desconocido: {espacio}')


def bordes(X, prof=2):
    """Extrae las 'prof' filas/columnas mas cercanas a cada borde de cada parche, ordenadas desde
    el borde hacia adentro. Devuelve dict con arreglos (N, L, prof, ch):
    'R' (derecha), 'L' (izquierda), 'B' (abajo), 'T' (arriba). 
    [:, :, 0] es la fila/columna del borde."""

    E = {'R': np.moveaxis(X[:, :, ::-1][:, :, :prof], 2, 2),
         'L': X[:, :, :prof],
         'B': np.moveaxis(X[:, ::-1][:, :prof], 1, 2),
         'T': np.moveaxis(X[:, :prof], 1, 2)}
    
    return {k: np.ascontiguousarray(v, dtype=np.float64) for k, v in E.items()}



# Medidas para una direccion. 
# EA: bordes del parche i (lado que mira a j) 
# EB: bordes del parche j (lado que mira a i) 
# ambos (N, L, K, ch) ordenados desde el borde hacia adentro.

def _ssd(EA, EB, g):
    N, L = EA.shape[:2]
    a, b = EA[:, :, 0].reshape(N, -1), EB[:, :, 0].reshape(N, -1)
    D = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T
    return np.maximum(D, 0) / L


def _pred(EA, EB, g, p=0.3, bloque=16):
    """Prediccion lineal: 
    b^ = A1 + g (A1 - A2) predice B1 
    a^ = B1 + g (B1 - B2) predice A1

    D = mean |B1 - b^|^p + mean |A1 - a^|^p (p < 1)."""

    N, L = EA.shape[:2]
    A1, A2, B1, B2 = EA[:, :, 0], EA[:, :, 1], EB[:, :, 0], EB[:, :, 1]

    pb = A1 + g * (A1 - A2) # prediccion de la columna de j desde i
    pa = B1 + g * (B1 - B2) # prediccion de la columna de i desde j
    D = np.empty((N, N))
    for i0 in range(0, N, bloque): # por bloques 
        i1 = min(N, i0 + bloque)
        r1 = np.abs(B1[None, :] - pb[i0:i1, None]) ** p # (b, N, L, ch)
        r2 = np.abs(A1[i0:i1, None] - pa[None, :]) ** p
        D[i0:i1] = (r1.sum(axis=(2, 3)) + r2.sum(axis=(2, 3))) / L
    return D


def _mgc_un_lado(EA, EB, g, lam, base, capas):
    """Termino de Mahalanobis visto desde el borde de i (para todos los pares i, j a la vez):

        D[i, j] = sum_l (G_ijl - mu_i)^T P_i (G_ijl - mu_i),   G_ijl = (B1_jl - A1_il) / g

    l recorre los pixeles del borde. G_ijl es el salto por pixel al cruzar de i a j.
    mu_i y S_i (P_i = S_i^-1) son la media y la covarianza 3x3 entre canales del gradiente
    del borde de i, medido dentro del parche como (A_t - A_(t+b)) / b, t = 0..capas-1.
    Por defecto b = 1 y capas = 1, es decir A1 - A2. Se suma lam * I a S_i para poder
    invertirla en bordes planos.

    Para evitar un bucle sobre los N^2 pares se expande el cuadrado:
        (v - u)^T P (v - u) = v^T P v - 2 u^T P v + u^T P u,   v = B1/g,  u = A1/g + mu_i
    v depende solo de j y u solo de i, asi que el termino cruzado es un producto de matrices."""

    N, L, K, ch = EA.shape
    A1, B1 = EA[:, :, 0], EB[:, :, 0]
    grads = np.concatenate([(EA[:, :, t] - EA[:, :, t + base]) / base for t in range(capas)], axis=1)
    mu = ((A1 - EA[:, :, base]) / base).mean(axis=1) # media en el borde (N, ch)
    dg = grads - grads.mean(axis=1, keepdims=True)
    S = np.einsum('nla,nlb->nab', dg, dg) / max(dg.shape[1] - 1, 1) + lam * np.eye(ch)
    P = np.linalg.inv(S)  # (N, ch, ch)
    u = A1 / g + mu[:, None] # (N, L, ch)
    v = B1 / g # (N, L, ch)
    M = np.einsum('nla,nlb->nab', v, v) # sum_l v v^T de cada j
    t1 = np.einsum('iab,jab->ij', P, M)  # sum_l v_j^T P_i v_j
    Pu = np.einsum('nab,nlb->nla', P, u) # P_i u_il
    t2 = Pu.reshape(N, -1) @ v.reshape(N, -1).T  # sum_l u_il^T P_i v_jl
    t3 = np.einsum('nla,nla->n', Pu, u) # sum_l u_il^T P_i u_il
    return np.maximum(t1 - 2 * t2 + t3[:, None], 0)


def _mgc(EA, EB, g, lam=1.0, base=1, capas=1):
    L = EA.shape[1]
    b = g if base == 'g' else int(base)
    D_i = _mgc_un_lado(EA, EB, g, lam, b, capas) # desde el borde de i
    D_j = _mgc_un_lado(EB, EA, g, lam, b, capas).T # desde el borde de j
    return (D_i + D_j) / L


def _cuadratica_separable(u, v, P):
    """sum_l (v_jl - u_il)^T P_i (v_jl - u_il) para todos los pares (i, j), con productos matriciales. 
    u, v: (N, L, ch)
    P: (N, ch, ch) (una matriz por pieza i)."""

    N = u.shape[0]
    M = np.einsum('nla,nlb->nab', v, v)
    t1 = np.einsum('iab,jab->ij', P, M)
    Pu = np.einsum('nab,nlb->nla', P, u)
    t2 = Pu.reshape(N, -1) @ v.reshape(N, -1).T
    t3 = np.einsum('nla,nla->n', Pu, u)
    return np.maximum(t1 - 2 * t2 + t3[:, None], 0)


def _mgcl(EA, EB, g, lam=1.0, mezcla=0.5):
    """Variante local: en vez de comparar el gradiente que cruza la union con la media del
    gradiente del borde, se compara fila a fila con el gradiente local esperado:

        m_l = (1 - mezcla) * gA_l + mezcla * gB_l

    gA: gradiente en el borde de i 
    gB: gradiente en el borde de j (ambos orientados de i a j).

    Sigue siendo separable: G_l - m_l = [B1/g - mezcla*gB]_j - [A1/g + (1-mezcla)*gA]_i.
    La covarianza (Mahalanobis) es la del gradiente de cada lado."""

    N, L, K, ch = EA.shape
    A1, A2, B1, B2 = EA[:, :, 0], EA[:, :, 1], EB[:, :, 0], EB[:, :, 1]
    gA = A1 - A2  # i -> j
    gB = B2 - B1 # i -> j (entrando a j)

    def prec(gr):
        dg = gr - gr.mean(axis=1, keepdims=True)
        return np.linalg.inv(np.einsum('nla,nlb->nab', dg, dg) / max(L - 1, 1) + lam * np.eye(ch))

    u = A1 / g + (1 - mezcla) * gA
    v = B1 / g - mezcla * gB
    D_i = _cuadratica_separable(u, v, prec(gA))
    D_j = _cuadratica_separable(v, u, prec(gB)).T
    return (D_i + D_j) / L


METODOS = {'ssd': _ssd, 'pred': _pred, 'mgc': _mgc, 'mgcl': _mgcl}


def disimilitudes(parches, d, metodo='mgc', espacio='lab', sigma=0.0, **kw):
    """Calcula las matrices de disimilitud de todas las uniones posibles.
    D_h[i, j]: costo de poner j a la derecha de i.   
    D_v[i, j]: costo de poner j debajo de i.
    La diagonal (pieza consigo misma) queda en infinito.
    Por eficiencia, solo las bandas de borde se convierten al espacio de color: 
    el costo no depende del tamaño del parche sino del largo de sus bordes."""

    g = d + 1
    base = g if kw.get('base') == 'g' else int(kw.get('base', 1))
    prof = max(2, base + kw.get('capas', 1))
    E = bordes(parches, prof) # uint8 -> float, (N, L, prof, 3)
    E = {k: _color_banda(v, espacio) for k, v in E.items()}

    if sigma > 0:
        E = {k: gaussian_filter1d(v, sigma, axis=1, mode='nearest') for k, v in E.items()}

    f = METODOS[metodo]
    D_h = f(E['R'], E['L'], g, **kw)
    D_v = f(E['B'], E['T'], g, **kw)
    np.fill_diagonal(D_h, np.inf)
    np.fill_diagonal(D_v, np.inf)
    return D_h, D_v


def _color_banda(v, espacio):
    "Transforma una banda de borde (N, L, K, 3) con valores 0...255 al espacio de color pedido"
    N, L, K, _ = v.shape
    x = transformar_color(np.clip(v, 0, 255).astype(np.uint8).reshape(N, L, K, 3), espacio)
    return x.astype(np.float64)
