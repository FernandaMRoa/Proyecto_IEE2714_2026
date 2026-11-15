"""Ensamblador automatico (seccion 3) y modo sin anclaje (bonus).

Se busca la permutacion P que minimiza la energia E(P) = suma de los costos de todas las
uniones horizontales y verticales, respetando el anclaje.

Etapas:
  1. Estimar d y extraer los parches (interleave.py).

  2. Costos: disimilitudes D_h, D_v de todos los pares (compatibilidad.py), normalizadas como
     (D + tau) / (2a mejor alternativa de ambas piezas + tau). Valen < 1 solo si las dos
     piezas se eligen mutuamente, lo que hace comparables piezas y direcciones.

  3. Armado por confianza en la grilla R x C: desde la pieza anclada, en cada
     paso se llena la celda libre con mayor confianza 1 - c1/c2 
     (c1, c2: costo medio de la 1a y 2a mejor pieza con los vecinos ya puestos). 
     Se agregan soluciones de marco flotante desde otras semillas, a las que se impone el anclaje.

  4. Busqueda local (descenso en E): intercambio de pares de piezas y desplazamiento ciclico
     de bloques rectangulares.

  5. Busqueda en vecindad grande: los segmentos de uniones elegidas mutuamente 
     se reubican rigidamente en el marco y el resto se rearma con 3 y 4.

  Se devuelve la solucion de menor energia.

Sin anclaje (bonus): mismas etapas, solo con marco flotante y sin celda fija.
No se usa ningun solver ni optimizador externo.
"""

import time
from functools import lru_cache
import numpy as np
from .compatibilidad import disimilitudes
from .interleave import estimar_interleave, extraer_parches
from .randomizador import armar_mosaico

# configuracion automatica por defecto (no se ajusta por imagen)
CONFIG_DEFECTO = {
    'metodo': 'mgc+pred', # medida de disimilitud ('mgc', 'pred', 'ssd', 'mgc+pred')
    'espacio': 'lab', # espacio de color de los descriptores
    'peso_pred': 0.5, # peso del termino de prediccion en 'mgc+pred'
    'normalizacion': 'segundo', # 'segundo' (razon al 2do mejor), 'mediana' o 'ninguna'
    'tau': 0.5, # regularizacion de la normalizacion (fraccion de la mediana)
    'suavizado_aditivo': True, # (D + tau) / (2do + tau): cuando ~0 quedan neutros (= 1)
    'refinar': True, # busqueda local (intercambios + bloques)
    'segmentos': True, # re-ensamblado por segmentos (vecindad grande)
    'semillas_extra': 6, # soluciones extra desde otras semillas (marco flotante)
    'max_iter_refinamiento': 50,
}

VECINOS = [(-1, 0, 'v', True), (1, 0, 'v', False), (0, -1, 'h', True), (0, 1, 'h', False)]
# (dr, dc, direccion, el vecino esta antes): ej. (-1, 0): el vecino esta arriba -> costo D_v[vecino, k]


# ----------------------------------------------------------------------------------------------
# 2. Costos normalizados

def _normalizar(D, modo, tau_rel=0.5, suavizado_aditivo=True):
    """Normaliza una matriz de disimilitud D[i, j] (i antes que j).

    'segundo': (D[i,j] + tau) / (min(2do mejor de la fila i, 2do mejor de la columna j) + tau).
               Mide cuan buena es la union comparada con la mejor alternativa de ambas piezas:
               < 1 solo si (i, j) es la mejor opcion de las dos. >= 1 en otro caso. 
               tau (una fraccion de la mediana de las mejores uniones) hace que los empates 
               entre piezas casi identicas (cielo, zonas saturadas o negras, D ~ 0) queden 
               neutros (~1) en vez de parecer uniones perfectas.

    'mediana': D / mediana global (solo hace comparables las dos direcciones)."""

    D = D.copy()
    f = np.isfinite(D)

    if modo == 'ninguna':
        return D
    
    if modo == 'mediana':
        return D / (np.median(D[f]) + 1e-12)
    
    if modo == 'segundo':
        fila2 = np.sort(D, axis=1)[:, 1]
        col2 = np.sort(D, axis=0)[1, :]
        esc = np.minimum(fila2[:, None], col2[None, :])
        # tau evita dividir por 0 cuando hay piezas casi identicas (cielo, regiones negras):
        # es una fraccion de la mediana de las mejores uniones de cada pieza
        tau = tau_rel * np.median(np.min(D, axis=1)) + 1e-12
        if suavizado_aditivo:
            return (D + tau) / (esc + tau)
        return D / (esc + tau)
    raise ValueError(modo)


def costos(parches, d, config):
    "Disimilitudes normalizadas Dn_h, Dn_v segun la configuracion"
    met = config['metodo']

    if met == 'mgc+pred':
        a_h, a_v = disimilitudes(parches, d, 'mgc', config['espacio'])
        b_h, b_v = disimilitudes(parches, d, 'pred', config['espacio'])
        med = lambda D: np.median(D[np.isfinite(D)]) + 1e-12
        D_h = a_h / med(a_h) + config['peso_pred'] * b_h / med(b_h)
        D_v = a_v / med(a_v) + config['peso_pred'] * b_v / med(b_v)
    else:
        D_h, D_v = disimilitudes(parches, d, met, config['espacio'])
    tau = config['tau']
    sa = config['suavizado_aditivo']
    return (_normalizar(D_h, config['normalizacion'], tau, sa),
            _normalizar(D_v, config['normalizacion'], tau, sa))


def energia(grid, Dh, Dv):
    "Suma de los costos de todas las uniones horizontales y verticales de la grilla"
    return float(Dh[grid[:, :-1], grid[:, 1:]].sum() + Dv[grid[:-1, :], grid[1:, :]].sum())


# ----------------------------------------------------------------------------------------------
# 3. Colocacion por confianza

def _costo_celda(grid, r, c, Dh, Dv):
    """Vector de costos (para todas las piezas) de ocupar la celda (r, c) dados sus vecinos ya
    colocados, y el numero de vecinos considerados."""
    H, W = grid.shape
    tot = 0.0
    n = 0
    for dr, dc, dire, antes in VECINOS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < H and 0 <= cc < W and grid[rr, cc] >= 0:
            v = grid[rr, cc]
            D = Dh if dire == 'h' else Dv
            tot = tot + (D[v, :] if antes else D[:, v])
            n += 1
    return tot, n


def colocar_codicioso(Dh, Dv, R, C, iniciales, flotante=False):
    """Colocacion codiciosa por confianza.

    iniciales : dict {(r, c): pieza} con las piezas ya ubicadas (anclaje, o un segmento completo).

    - Marco fijo (flotante=False): grilla R x C, las iniciales tienen posicion absoluta.
    - Marco flotante (flotante=True, una sola semilla): grilla (2R-1) x (2C-1) con la semilla al
      centro. Solo se aceptan celdas que mantengan la caja de piezas dentro de R x C.

    En cada paso se evaluan las celdas libres con al menos un vecino colocado (frontera).
    Para cada una se ordenan las piezas libres por costo medio con sus vecinos y se calcula la
    confianza 1 - costo(1ra)/costo(2da). Se coloca la mejor pieza de la celda mas confiable
    (a igualdad, la de mas vecinos). La evaluacion de cada celda se guarda y solo se recalcula
    si cambia un vecino o si se uso alguna de sus dos mejores piezas.

    Devuelve la grilla R x C con el indice (0-based) de la pieza en cada celda."""
    N = R * C
    if flotante:
        H, W = 2 * R - 1, 2 * C - 1
        (k0,) = iniciales.values()
        iniciales = {(R - 1, C - 1): k0}
    else:
        H, W = R, C
    grid = -np.ones((H, W), dtype=int)
    libre = np.ones(N, dtype=bool)
    for (r, c), k in iniciales.items():
        grid[r, c] = k
        libre[k] = False
    pos = np.array(list(iniciales.keys()))
    caja = [pos[:, 0].min(), pos[:, 0].max(), pos[:, 1].min(), pos[:, 1].max()]

    def evaluar(r, c):
        tot, n = _costo_celda(grid, r, c, Dh, Dv)
        cand = np.where(libre, tot / n, np.inf)
        if libre.sum() >= 2:
            k1, k2 = np.argpartition(cand, 1)[:2]
            if cand[k2] < cand[k1]:
                k1, k2 = k2, k1
            conf = 1.0 - cand[k1] / (cand[k2] + 1e-12)
        else:
            k1 = k2 = int(np.argmin(cand))
            conf = 1.0
        return (conf, n), int(k1), int(k2)

    def valida(r, c):
        if not (0 <= r < H and 0 <= c < W) or grid[r, c] >= 0:
            return False
        if flotante and (max(caja[1], r) - min(caja[0], r) >= R or
                         max(caja[3], c) - min(caja[2], c) >= C):
            return False
        return True

    frontera = {}
    for (r, c) in iniciales:
        for dr, dc, _, _ in VECINOS:
            if valida(r + dr, c + dc):
                frontera[(r + dr, c + dc)] = None

    for _ in range(int(libre.sum())):
        if flotante: # la caja pudo haber crecido -> sacar celdas que ya no caben
            for cel in [p for p in frontera if not valida(*p)]:
                del frontera[cel]
        mejor = None
        for cel, ev in frontera.items():
            if ev is None:
                ev = frontera[cel] = evaluar(*cel)
            if mejor is None or ev[0] > mejor[1][0]:
                mejor = (cel, ev)
        (r, c), (_, k, _) = mejor
        grid[r, c] = k
        libre[k] = False
        caja = [min(caja[0], r), max(caja[1], r), min(caja[2], c), max(caja[3], c)]
        del frontera[(r, c)]
        for cel, ev in frontera.items(): # su mejor o 2da pieza ya no esta libre
            if ev is not None and (ev[1] == k or ev[2] == k):
                frontera[cel] = None
        for dr, dc, _, _ in VECINOS: # vecinos nuevos o con un vecino mas
            if valida(r + dr, c + dc):
                frontera[(r + dr, c + dc)] = None
    if flotante:
        grid = grid[caja[0]:caja[0] + R, caja[2]:caja[2] + C]
    return grid


# ----------------------------------------------------------------------------------------------
# 4. Refinamiento local

def _matriz_costo_local(grid, Dh, Dv):
    """L[p, k] = costo de poner la pieza k en la celda p (fila mayor) con los vecinos actuales."""
    R, C = grid.shape
    N = Dh.shape[0]
    L = np.zeros((R, C, N))
    L[:, 1:] += Dh[grid[:, :-1], :] # vecino izquierdo v: D_h[v, k]
    L[:, :-1] += Dh[:, grid[:, 1:]].transpose(1, 2, 0) # vecino derecho v: D_h[k, v]
    L[1:, :] += Dv[grid[:-1, :], :] # vecino de arriba
    L[:-1, :] += Dv[:, grid[1:, :]].transpose(1, 2, 0) # vecino de abajo
    return L.reshape(R * C, N)


def _mejor_intercambio(grid, Dh, Dv, movibles):
    """Ganancia de energia de intercambiar cada par de celdas (vectorizado). 
    Para celdas no vecinas: G[p,q] = L[p,a_p] + L[q,a_q] - L[p,a_q] - L[q,a_p]. 
    Para celdas vecinas la union comun cambia de sentido y se evalua exactamente."""

    R, C = grid.shape
    flat = grid.ravel()
    L = _matriz_costo_local(grid, Dh, Dv)
    Lp = L[:, flat] # Lp[p, q] = costo de la pieza de q puesta en p
    dg = np.diag(Lp)
    G = dg[:, None] + dg[None, :] - Lp - Lp.T
    mask = np.zeros_like(G, dtype=bool)
    mask[np.ix_(movibles, movibles)] = True
    np.fill_diagonal(mask, False)
    E = energia(grid, Dh, Dv)

    for p in movibles:  # pares vecinos: calculo exacto
        r, c = divmod(p, C)
        for q in (p + 1 if c < C - 1 else None, p + C if r < R - 1 else None):
            if q is None or not mask[p, q]:
                continue
            g2 = flat.copy()
            g2[p], g2[q] = g2[q], g2[p]
            G[p, q] = G[q, p] = E - energia(g2.reshape(R, C), Dh, Dv)
    G[~mask] = -np.inf
    p, q = np.unravel_index(np.argmax(G), G.shape)
    return G[p, q], p, q


def energias_lote(grids, Dh, Dv):
    "Energia de un lote de celdas (M, R, C), vectorizada"
    return (Dh[grids[:, :, :-1], grids[:, :, 1:]].sum(axis=(1, 2)) +
            Dv[grids[:, :-1, :], grids[:, 1:, :]].sum(axis=(1, 2)))


@lru_cache(maxsize=64)
def _permutaciones_bloque(R, C, fija):
    """Indices de todos los desplazamientos ciclicos de bloques rectangulares (que no contengan la
    celda fija) en k celdas a lo largo de filas o columnas (k = 1...alto-1 o 1...ancho-1).
    Cada fila de la matriz devuelta es una permutacion de las R*C celdas: grid_nueva = grid[perm].
    Un bloque armado correctamente pero corrido queda en su lugar y la franja desplazada pasa al
    otro lado del bloque. Se calculan una sola vez por tamaño de grilla."""

    base = np.arange(R * C).reshape(R, C)
    perms = []
    for r0 in range(R):
        for r1 in range(r0 + 1, R + 1):
            for c0 in range(C):
                for c1 in range(c0 + 1, C + 1):
                    if fija is not None and r0 <= fija[0] < r1 and c0 <= fija[1] < c1:
                        continue
                    sub = base[r0:r1, c0:c1]
                    for eje, n in ((0, r1 - r0), (1, c1 - c0)):
                        for k in range(1, n):
                            g2 = base.copy()
                            g2[r0:r1, c0:c1] = np.roll(sub, k, axis=eje)
                            perms.append(g2.ravel())
    return np.array(perms, dtype=np.int32).reshape(-1, R * C)


def _mejor_bloque(grid, Dh, Dv, fijas):
    "Mejor desplazamiento ciclico de bloque (None si ninguno baja la energia)"
    R, C = grid.shape
    fija = tuple(fijas[0]) if fijas else None
    perms = _permutaciones_bloque(R, C, fija)
    if len(perms) == 0:
        return None, None
    cands = grid.ravel()[perms].reshape(-1, R, C)
    E = energias_lote(cands, Dh, Dv)
    k = int(np.argmin(E))
    if E[k] < energia(grid, Dh, Dv) - 1e-9:
        return float(E[k]), cands[k].copy()
    return None, None


def refinar(grid, Dh, Dv, fijas=(), max_iter=50):
    """Busqueda local de descenso sobre la energia global. Se aplican intercambios de pares
    (el de mayor ganancia) hasta que no haya ganancia. Luego se prueba el mejor desplazamiento
    ciclico de bloque y, si baja la energia, se vuelve a los intercambios. 
    Las celdas 'fijas' (anclaje) no se mueven."""

    grid = grid.copy()
    R, C = grid.shape
    fijas = [tuple(f) for f in fijas]
    movibles = [r * C + c for r in range(R) for c in range(C) if (r, c) not in fijas]
    hist = [energia(grid, Dh, Dv)]
    for _ in range(max_iter):
        while True:
            gan, p, q = _mejor_intercambio(grid, Dh, Dv, movibles)
            if not gan > 1e-9:
                break
            f = grid.ravel()
            f[p], f[q] = f[q], f[p]
            hist.append(energia(grid, Dh, Dv))
            if len(hist) > 10 * R * C:
                break
        E2, g2 = _mejor_bloque(grid, Dh, Dv, fijas)
        if g2 is None:
            break
        grid = g2
        hist.append(E2)
    return grid, hist


def segmentos(grid, Dh, Dv):
    """Divide la solucion en segmentos: grupos de celdas conectadas por uniones de "mejores
    amigos" (a y b son mejores amigos si b es la mejor pieza para ir junto a a, y a la mejor
    para ir junto a b, en esa direccion). Son las zonas armadas con mas seguridad.
    Se agrupan con union-find. Devuelve listas de celdas (r, c), del segmento mas grande al mas chico."""
    R, C = grid.shape
    der, izq = np.argmin(Dh, axis=1), np.argmin(Dh, axis=0)
    aba, arr = np.argmin(Dv, axis=1), np.argmin(Dv, axis=0)
    padre = list(range(R * C))

    def raiz(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x
    for r in range(R):
        for c in range(C):
            a = grid[r, c]
            if c + 1 < C and der[a] == grid[r, c + 1] and izq[grid[r, c + 1]] == a:
                padre[raiz(r * C + c)] = raiz(r * C + c + 1)
            if r + 1 < R and aba[a] == grid[r + 1, c] and arr[grid[r + 1, c]] == a:
                padre[raiz(r * C + c)] = raiz((r + 1) * C + c)
    grupos = {}
    for p in range(R * C):
        grupos.setdefault(raiz(p), []).append(divmod(p, C))
    return sorted(grupos.values(), key=len, reverse=True)


def reensamblar_segmentos(grid, Dh, Dv, anclaje=None, n_segmentos=4, n_refinar=6, max_rondas=6, max_iter=50):
    """Busqueda en vecindad grande: se toma cada uno de los segmentos mas grandes de la solucion
    actual, se prueba ubicarlo fijamente en todas las posiciones posibles del marco (respetando
    el anclaje), se completa el resto con la colocacion codiciosa y se refinan las mejores
    opciones. Si alguna baja la energia se acepta y se repite. 
    Corrige soluciones formadas por bloques correctos mal ubicados entre si, que la busqueda local no puede mover."""
    R, C = grid.shape
    fijas = [anclaje[0]] if anclaje else []
    mejor, E_mejor = grid.copy(), energia(grid, Dh, Dv)
    for _ in range(max_rondas):
        mejoro = False
        segs = [s for s in segmentos(mejor, Dh, Dv) if 2 <= len(s) < R * C][:n_segmentos]
        for seg in segs:
            celdas = np.array(seg)
            piezas = [int(mejor[r, c]) for r, c in seg]
            rmin, cmin = celdas.min(axis=0)
            rel = celdas - [rmin, cmin]
            h, w = rel.max(axis=0) + 1
            opciones = []
            for orow in range(R - h + 1):
                for ocol in range(C - w + 1):
                    ini = {(int(r + orow), int(c + ocol)): k for (r, c), k in zip(rel, piezas)}
                    if anclaje is not None:
                        pos_a, k_a = anclaje
                        if k_a in piezas:
                            if ini.get(pos_a) != k_a:
                                continue
                        elif pos_a in ini:
                            continue
                        else:
                            ini[pos_a] = k_a
                    g = colocar_codicioso(Dh, Dv, R, C, ini)
                    opciones.append((energia(g, Dh, Dv), g))
            opciones.sort(key=lambda x: x[0])
            for _, g in opciones[:n_refinar]:
                g, _ = refinar(g, Dh, Dv, fijas, max_iter)
                E = energia(g, Dh, Dv)
                if E < E_mejor - 1e-9:
                    mejor, E_mejor, mejoro = g, E, True
        if not mejoro:
            break
    return mejor


# ----------------------------------------------------------------------------------------------
# 5. Semillas para el marco flotante

def _semillas(Dh, Dv, n):
    """Piezas con mas 'mejores amigos' (i es la mejor opcion de j y j la de i, en la direccion correspondiente). 
    Son las zonas mas confiables para empezar."""
    bb = np.zeros(Dh.shape[0])
    for D in (Dh, Dv):
        der = np.argmin(D, axis=1) # mejor vecino siguiente de cada i
        izq = np.argmin(D, axis=0) # mejor vecino anterior de cada j
        for i, j in enumerate(der):
            if izq[j] == i:
                bb[i] += 1
                bb[j] += 1
    orden = np.argsort(-bb, kind='stable')
    return [int(k) for k in orden[:n]], bb


# ----------------------------------------------------------------------------------------------
# Punto de entrada

def ensamblar(mosaico, R, C, anclaje=None, config=None, d=None):
    """Reconstruye el mosaico.

    mosaico : RGB uint8 H x W
    R, C    : filas y columnas de la grilla
    anclaje : (rs, cs, rt, ct) en indices 1-based (la pieza de la celda de entrada (rs, cs) va en
              la celda de salida (rt, ct)). None -> modo sin anclaje (bonus).
    config  : dict que modifica CONFIG_DEFECTO (opcional)
    d       : interleave conocido (solo para experimentos, por defecto se estima)

    Devuelve dict con: 'imagen' (reconstruida, mismo tamaño), 'P' (R x C, k 1-based),
    'd', 'tiempos', 'diagnostico'."""

    cfg = dict(CONFIG_DEFECTO)
    if config:
        cfg.update(config)
    t = {}
    t0 = time.perf_counter()

    diag_d = None
    if d is None:
        d, diag_d = estimar_interleave(mosaico, R, C)
    parches, geo = extraer_parches(mosaico, R, C, d)
    t['interleave_y_extraccion'] = time.perf_counter() - t0

    t1 = time.perf_counter()
    Dh, Dv = costos(parches, d, cfg)
    t['compatibilidad'] = time.perf_counter() - t1

    t2 = time.perf_counter()
    candidatos = []

    if anclaje is not None:
        rs, cs, rt, ct = anclaje
        pieza_anc = (rs - 1) * C + (cs - 1)
        pos_anc = (rt - 1, ct - 1)
        candidatos.append(('anclaje', colocar_codicioso(Dh, Dv, R, C, {pos_anc: pieza_anc})))
    semillas, bb = _semillas(Dh, Dv, cfg['semillas_extra'] if anclaje is not None
                             else max(cfg['semillas_extra'], 1))
    for s in semillas:
        g = colocar_codicioso(Dh, Dv, R, C, {(0, 0): s}, flotante=True)
        if anclaje is not None and g[pos_anc] != pieza_anc:
            # la solucion flotante no respeta el anclaje: se refina libre y luego se fuerza el
            # anclaje intercambiando la pieza anclada con la que ocupa su celda
            if cfg['refinar']:
                g, _ = refinar(g, Dh, Dv, [], cfg['max_iter_refinamiento'])
            if g[pos_anc] != pieza_anc:
                r_a, c_a = np.argwhere(g == pieza_anc)[0]
                g[r_a, c_a], g[pos_anc] = g[pos_anc], pieza_anc
        candidatos.append((f'flotante_{s}', g))
    t['colocacion'] = time.perf_counter() - t2

    t3 = time.perf_counter()
    fijas = [pos_anc] if anclaje is not None else []
    resultados = []
    for nombre, g in candidatos:
        E0 = energia(g, Dh, Dv)
        if cfg['refinar']:
            g, hist = refinar(g, Dh, Dv, fijas, cfg['max_iter_refinamiento'])
        resultados.append((energia(g, Dh, Dv), nombre, g, E0))
    resultados.sort(key=lambda x: x[0])
    E, nombre, grid, E0 = resultados[0]
    if cfg['segmentos']:
        anc = (pos_anc, pieza_anc) if anclaje is not None else None
        grid = reensamblar_segmentos(grid, Dh, Dv, anc, max_iter=cfg['max_iter_refinamiento'])
        if energia(grid, Dh, Dv) < E - 1e-9:
            nombre += '+segmentos'
        E = energia(grid, Dh, Dv)
    t['refinamiento'] = time.perf_counter() - t3

    imagen = armar_mosaico(parches, geo, orden=grid.ravel())
    P = grid + 1 # k 1-based: k = (r_s - 1) C + c_s
    t['total'] = time.perf_counter() - t0

    diag = {'interleave': diag_d, 'config': cfg, 'energia_final': E, 'energia_codiciosa': E0,
            'solucion_elegida': nombre,
            'candidatos': [{'nombre': n, 'energia_inicial': e0, 'energia_final': e}
                           for e, n, _, e0 in resultados],
            'geometria': geo, 'best_buddies_por_pieza': bb.tolist()}
    
    return {'imagen': imagen, 'P': P, 'd': int(d), 'tiempos': t, 'diagnostico': diag}
