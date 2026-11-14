"""Randomizador de parches (script de linea de comandos).

Toma una imagen RGB cualquiera, la corta en una grilla de R x C parches separados por franjas
de ceros de ancho d, desordena los parches al azar y guarda el resultado como un caso de prueba
para el ensamblador.

Uso:
    python randomizar.py IMAGEN R C D SEMILLA --salida CARPETA

    IMAGEN    ruta de la imagen original
    R, C      numero de filas y columnas de la grilla
    D         ancho de las franjas de ceros (interleave): 0, 1, 3 o 5
    SEMILLA   entero. Con la misma semilla se obtiene exactamente el mismo caso
    CARPETA   donde se guardan los archivos (por defecto 'caso')

Ejemplo:
    python randomizar.py imagenes/coffee.png 5 7 3 0 --salida caso

Archivos que genera:
    CARPETA/mosaico.png         imagen desordenada: es la entrada del ensamblador.
                                Se guarda en PNG para que los ceros de las
                                franjas sigan siendo exactamente cero.

    CARPETA/caso.json           informacion publica que si recibe el ensamblador:
                                R, C y el anclaje (rs, cs, rt, ct), con indices 1-based.

    CARPETA/gt/gt.json          ground truth: orden correcto P_GT, d, semilla y geometria.
                                El ensamblador nunca lo lee; solo lo usa evaluar.py, despues
                                de ensamblar, para calcular A_pos y A_adj.

    CARPETA/gt/referencia.png   imagen correcta (crop con las franjas en cero), para comparar
                                visualmente con la reconstruccion.

Si la configuracion no cumple los rangos oficiales el script la rechaza y muestra el motivo.
"""

import argparse
import json
import os
import sys
from mosaico.imagenes import leer_rgb, guardar_rgb
from mosaico.randomizador import generar_caso, referencia


def main():
    ap = argparse.ArgumentParser(description='Genera un mosaico desordenado y su ground truth')
    ap.add_argument('imagen')
    ap.add_argument('R', type=int)
    ap.add_argument('C', type=int)
    ap.add_argument('d', type=int, help='interleave (0, 1, 3 o 5)')
    ap.add_argument('semilla', type=int)
    ap.add_argument('--salida', default='caso')
    ap.add_argument('--no-estricto', action='store_true',
                    help='permite configuraciones fuera de los rangos oficiales (solo pruebas)')
    a = ap.parse_args()

    img = leer_rgb(a.imagen)
    try:
        mosaico, publico, gt = generar_caso(img, a.R, a.C, a.d, a.semilla, estricto=not a.no_estricto)
    except ValueError as e:
        sys.exit(f'Configuracion rechazada: {e}')

    publico['imagen'] = 'mosaico.png'
    gt['imagen_original'] = os.path.abspath(a.imagen)

    os.makedirs(os.path.join(a.salida, 'gt'), exist_ok=True)
    guardar_rgb(os.path.join(a.salida, 'mosaico.png'), mosaico)

    with open(os.path.join(a.salida, 'caso.json'), 'w') as f:
        json.dump(publico, f, indent=2)

    with open(os.path.join(a.salida, 'gt', 'gt.json'), 'w') as f:
        json.dump(gt, f, indent=2)
        
    guardar_rgb(os.path.join(a.salida, 'gt', 'referencia.png'), referencia(img, a.R, a.C, a.d))
    g = gt['geometria']

    print(f'Mosaico {g["H"]}x{g["W"]}, grilla {a.R}x{a.C}, parches {g["p_h"]}x{g["p_w"]}, d={a.d}')
    print(f'Anclaje (rs, cs) -> (rt, ct) = {tuple(publico["anclaje"][:2])} -> {tuple(publico["anclaje"][2:])}')
    print(f'Archivos en {a.salida}/ (ground truth separado en {a.salida}/gt/)')


if __name__ == '__main__':
    main()
