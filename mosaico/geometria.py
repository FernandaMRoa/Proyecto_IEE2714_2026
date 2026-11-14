"""Geometria de la grilla de parches (convencion del enunciado, seccion 2).

Para una imagen original de H0 x W0, una grilla R x C y un interleave d:
    s_h = floor(H0 / R),  s_w = floor(W0 / C)      (tamaño de cada celda)
    p_h = s_h - d,        p_w = s_w - d            (tamaño util de cada parche)
    q   = floor(d / 2)                             (ceros antes del parche dentro de la celda)
Dentro de cada celda el parche empieza q pixeles despues del borde superior/izquierdo, y quedan
q + 1 ceros despues de el (si d > 0). Entre dos parches vecinos hay exactamente d ceros.

Convencion de indices en todo el codigo:
    - internamente se usan indices 0..R-1, 0..C-1 y parches numerados k0 = r*C + c (0..R*C-1)
    - hacia el usuario (CLI, archivos JSON) se usan indices 1..R, 1..C y k = (r-1)*C + c (1..R*C),
      igual que en el enunciado.
"""

INTERLEAVES_VALIDOS = (0, 1, 3, 5)
RANGO_GRILLA = (4, 9)        # R y C en los casos oficiales
RANGO_PARCHE = (64, 512)     # alto y ancho util de los casos oficiales


def offset_q(d):
    "Numero de ceros antes del parche dentro de su celda (q = floor(d/2))"
    return d // 2


def geometria(H0, W0, R, C, d, estricto=True):
    """Calcula la geometria de la grilla y la valida.
    Con estricto=True se exige cumplir con los rangos oficiales 
    (4 <= R,C <= 9, parches de 64 a 512 px, d en {0,1,3,5}). 
    Si algo no se cumple retorna ValueError."""

    if d not in INTERLEAVES_VALIDOS and estricto:
        raise ValueError(f'interleave d={d} no permitido (validos: {INTERLEAVES_VALIDOS})')
    
    if d < 0:
        raise ValueError('d debe ser >= 0')
    
    if R < 1 or C < 1:
        raise ValueError('R y C deben ser positivos')
    
    s_h, s_w = H0 // R, W0 // C
    p_h, p_w = s_h - d, s_w - d

    if p_h < 2 or p_w < 2:
        raise ValueError(f'parches demasiado pequeños ({p_h}x{p_w})')
    
    if estricto:
        lo, hi = RANGO_GRILLA
        
        if not (lo <= R <= hi and lo <= C <= hi):
            raise ValueError(f'R={R}, C={C} fuera del rango oficial [{lo}, {hi}]')
        
        lo, hi = RANGO_PARCHE

        if not (lo <= p_h <= hi and lo <= p_w <= hi):
            raise ValueError(f'parche util {p_h}x{p_w} fuera del rango oficial [{lo}, {hi}] '
                             f'(imagen {H0}x{W0}, grilla {R}x{C}, d={d})')

    return {'R': R, 'C': C, 'd': d, 'q': offset_q(d), 's_h': s_h, 's_w': s_w,
            'p_h': p_h, 'p_w': p_w, 'H': R * s_h, 'W': C * s_w}


def filas_separacion_locales(s, d):
    """Indices locales (dentro de una celda de largo s) que son separacion para un interleave d.
    q ceros al inicio de la celda y q+1 al final (d = 2q+1). Para d = 0 no hay separacion."""
    if d == 0:
        return []
    q = offset_q(d)
    return list(range(0, q)) + list(range(s - (q + 1), s))


def k_usuario(r0, c0, C):
    "Indice 1-based del enunciado, k = (r-1)C + c, a partir de indices 0-based"
    return r0 * C + c0 + 1


def rc_desde_k(k, C):
    "Inverso de k_usuario: devuelve (r0, c0) 0-based"
    return (k - 1) // C, (k - 1) % C
