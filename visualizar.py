"""Visualizacion continua de una reconstruccion (script de linea de comandos).

Rellena las franjas de separacion de una imagen reconstruida para verla "continua".

Uso:
    python visualizar.py RECONSTRUIDA R C [--d D] [--metodo METODO] [--salida ARCHIVO]

    RECONSTRUIDA   imagen con los parches ordenados y franjas en cero (reconstruida.png)
    R, C           filas y columnas de la grilla
    --d            interleave. Si no se da, se estima igual que en el ensamblador
    --metodo       lineal (por defecto): lineal en franjas y bilineal en cruces
                   cubico: Hermite, transicion mas suave (mejor con d = 1)
                   armonico: ecuacion de Laplace (mejor con d >= 3)
    --salida       archivo de salida (por defecto visualizacion.png)

Ejemplo:
    python visualizar.py salida/reconstruida.png 5 7 --metodo armonico --salida salida/vis.png

El borde externo de la imagen se rellena repitiendo el valor del borde.
"""
import argparse
from mosaico.imagenes import leer_rgb, guardar_rgb
from mosaico.interleave import estimar_interleave
from mosaico.visualizacion import rellenar_separaciones


def main():
    ap = argparse.ArgumentParser(description='Interpola las franjas de separacion')
    ap.add_argument('reconstruida')
    ap.add_argument('R', type=int)
    ap.add_argument('C', type=int)
    ap.add_argument('--d', type=int)
    ap.add_argument('--metodo', default='lineal', choices=['lineal', 'cubico', 'armonico'])
    ap.add_argument('--salida', default='visualizacion.png')
    a = ap.parse_args()
    img = leer_rgb(a.reconstruida)
    d = a.d if a.d is not None else estimar_interleave(img, a.R, a.C)[0]
    guardar_rgb(a.salida, rellenar_separaciones(img, a.R, a.C, d, a.metodo))
    print(f'd = {d}, metodo = {a.metodo} -> {a.salida}')


if __name__ == '__main__':
    main()
