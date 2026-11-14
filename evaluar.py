"""Evaluacion con ground truth (script de linea de comandos).

Compara el orden que encontro el ensamblador con el orden correcto guardado por el
randomizador y calcula las metricas del enunciado. Es una rutina externa: se ejecuta despues
del ensamblado, y es el unico script que lee el ground truth.

Uso:
    python evaluar.py RESULTADO GT

    RESULTADO   salida del ensamblador: resultado.json (o P_estimada.txt)
    GT          ground truth del randomizador: CARPETA/gt/gt.json

Ejemplo:
    python evaluar.py salida/resultado.json caso/gt/gt.json

Metricas que calcula:
    A_pos       fraccion de piezas ubicadas en su celda correcta

    A_adj       fraccion de adyacencias verdaderas (izquierda->derecha y arriba->abajo) que
                aparecen en la reconstruccion, aunque el par este desplazado

    perfecta    True si A_pos = 1 (todas las piezas en su lugar)
    Ademas: numero de piezas correctas, total de piezas y total de adyacencias, y, si
    RESULTADO es resultado.json, si el interleave d se estimo correctamente.

Las metricas se muestran en pantalla y se guardan en evaluacion.json, en la misma carpeta
de RESULTADO.
"""

import argparse
import json
import os
import numpy as np
from mosaico.evaluacion import metricas

def leer_P(ruta):
    if ruta.endswith('.json'):
        with open(ruta) as f:
            return np.array(json.load(f)['P'])
    return np.loadtxt(ruta, dtype=int, ndmin=2)


def main():
    ap = argparse.ArgumentParser(description='Compara P estimada con el ground truth')
    ap.add_argument('resultado')
    ap.add_argument('gt')
    a = ap.parse_args()
    P = leer_P(a.resultado)
    with open(a.gt) as f:
        gt = json.load(f)
    m = metricas(P, gt['P_gt'])
    if a.resultado.endswith('.json'):
        with open(a.resultado) as f:
            m['d_estimado_correcto'] = json.load(f).get('d_estimado') == gt['d']
    print(json.dumps(m, indent=2))
    with open(os.path.join(os.path.dirname(os.path.abspath(a.resultado)), 'evaluacion.json'), 'w') as f:
        json.dump(m, f, indent=2)


if __name__ == '__main__':
    main()
