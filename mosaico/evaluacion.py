"""Metricas de evaluacion (seccion 5). Solo se usan despues del ensamblado, con el ground truth
leido desde un archivo externo."""

import numpy as np

def adyacencias(P):
    """Conjunto de adyacencias de una matriz de orden P (R x C):
    ('h', a, b) si la pieza b esta a la derecha de a. ('v', a, b) si b esta debajo de a."""

    P = np.asarray(P)
    R, C = P.shape
    ady = set()

    for r in range(R):
        for c in range(C - 1):
            ady.add(('h', int(P[r, c]), int(P[r, c + 1])))

    for r in range(R - 1):
        for c in range(C):
            ady.add(('v', int(P[r, c]), int(P[r + 1, c])))

    return ady


def metricas(P_est, P_gt):
    """A_pos = fraccion de parches en su celda correcta.
    A_adj = fraccion de adyacencias verdaderas (izq->der y arriba->abajo) recuperadas en P_est."""
    
    P_est, P_gt = np.asarray(P_est), np.asarray(P_gt)
    
    if P_est.shape != P_gt.shape:
        raise ValueError(f'P_est {P_est.shape} y P_gt {P_gt.shape} no tienen la misma forma')
    
    R, C = P_gt.shape
    a_pos = float(np.mean(P_est == P_gt))
    n_adj = R * (C - 1) + (R - 1) * C
    a_adj = len(adyacencias(P_est) & adyacencias(P_gt)) / n_adj

    return {'A_pos': a_pos, 'A_adj': float(a_adj), 'perfecta': bool(a_pos == 1.0),
            'n_correctas': int(np.sum(P_est == P_gt)), 'N': int(R * C), 'N_adj': int(n_adj)}
