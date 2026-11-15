"""Ensamblador automatico (script de linea de comandos).

Reconstruye un mosaico desordenado. 
Recibe solo lo que permite el enunciado: la imagen, R, C y el anclaje. 
El interleave d se estima solo y no hay parametros que ajustar por imagen.

Uso:
    python ensamblar.py MOSAICO R C RS CS RT CT --salida CARPETA

    MOSAICO        imagen desordenada
    R, C           filas y columnas de la grilla
    RS CS RT CT    anclaje, indices 1-based: la pieza de la celda (RS, CS) del mosaico va en la
                   celda (RT, CT) de la imagen correcta
    --salida       carpeta de resultados (por defecto 'salida')

Ejemplo:
    python ensamblar.py caso/mosaico.png 5 7 4 6 3 4 --salida salida

Opciones:
    --caso caso.json      toma R, C y el anclaje del archivo publico del randomizador
    --sin-anclaje         modo bonus: no usa el anclaje (RS CS RT CT se pueden omitir)
    --visualizar METODO   guarda ademas la imagen con las franjas rellenadas (lineal, cubico o armonico)
    --config JSON         cambia la configuracion por defecto, ej. '{"metodo": "mgc"}'

Salidas en CARPETA:
    reconstruida.png      imagen reordenada, mismo tamano que la entrada (franjas en cero)
    P_estimada.txt        matriz P de R x C: en cada celda, el indice k = (r_s - 1) * C + c_s
                          de la pieza del mosaico que se puso ahi
    resultado.json        P, d estimado, tiempos por etapa y diagnostico
    visualizacion.png     solo con --visualizar
"""
import argparse
import json
import os
import sys
import time
import numpy as np
from mosaico.imagenes import leer_rgb, guardar_rgb
from mosaico.ensamblador import ensamblar
from mosaico.visualizacion import rellenar_separaciones


def _a_json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main():
    ap = argparse.ArgumentParser(description='Reconstruye un mosaico de parches permutados')
    ap.add_argument('mosaico')
    ap.add_argument('R', type=int, nargs='?')
    ap.add_argument('C', type=int, nargs='?')
    ap.add_argument('anclaje', type=int, nargs='*', help='RS CS RT CT (1-based)')
    ap.add_argument('--caso', help='caso.json con R, C y anclaje')
    ap.add_argument('--sin-anclaje', action='store_true', help='modo bonus sin anclaje')
    ap.add_argument('--salida', default='salida')
    ap.add_argument('--visualizar', choices=['lineal', 'cubico', 'armonico'])
    ap.add_argument('--config', help='JSON con opciones del ensamblador')
    a = ap.parse_args()

    R, C, anc = a.R, a.C, a.anclaje
    if a.caso:
        with open(a.caso) as f:
            caso = json.load(f)
        R, C, anc = caso['R'], caso['C'], caso.get('anclaje')
    if R is None or C is None:
        sys.exit('Faltan R y C')
    if a.sin_anclaje:
        anc = None
    elif not anc or len(anc) != 4:
        sys.exit('Se requiere el anclaje RS CS RT CT (o --sin-anclaje)')
    else:
        for v, lim in zip(anc, (R, C, R, C)):
            if not 1 <= v <= lim:
                sys.exit(f'Anclaje {anc} fuera de la grilla {R}x{C} (indices 1-based)')
    cfg = json.loads(a.config) if a.config else None

    t0 = time.perf_counter()
    mosaico = leer_rgb(a.mosaico)
    t_lectura = time.perf_counter() - t0
    res = ensamblar(mosaico, R, C, anc, cfg)
    res['tiempos']['lectura'] = t_lectura

    os.makedirs(a.salida, exist_ok=True)
    guardar_rgb(os.path.join(a.salida, 'reconstruida.png'), res['imagen'])
    np.savetxt(os.path.join(a.salida, 'P_estimada.txt'), res['P'], fmt='%d')
    if a.visualizar:
        t1 = time.perf_counter()
        vis = rellenar_separaciones(res['imagen'], R, C, res['d'], a.visualizar)
        res['tiempos']['visualizacion'] = time.perf_counter() - t1
        guardar_rgb(os.path.join(a.salida, 'visualizacion.png'), vis)
        
    salida = {'R': R, 'C': C, 'anclaje': anc, 'd_estimado': res['d'], 'P': res['P'].tolist(),
              'tiempos_s': res['tiempos'], 'diagnostico': res['diagnostico']}
    with open(os.path.join(a.salida, 'resultado.json'), 'w') as f:
        json.dump(salida, f, indent=2, default=_a_json)

    print(f'd estimado = {res["d"]}  ({res["diagnostico"]["interleave"]["modo"]})')
    print('P estimada (k = (r_s-1)*C + c_s):')
    print(res['P'])
    print('Tiempos [s]: ' + ', '.join(f'{k}={v:.3f}' for k, v in res['tiempos'].items()))
    print(f'Resultados en {a.salida}/')


if __name__ == '__main__':
    main()
