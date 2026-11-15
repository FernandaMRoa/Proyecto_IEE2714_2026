"""Demostracion del flujo completo (script de linea de comandos).

Uso:
    python demo.py [IMAGEN] [R] [C] [D] [SEMILLA]

Todos los argumentos son opcionales. Por defecto: imagenes/coffee.png 5 7 3 0.
Ejecuta en orden los scripts del proyecto y guarda todo en demo/:
  1. randomizar.py -> mosaico desordenado (demo/mosaico.png), datos publicos (demo/caso.json)
                      y ground truth separado (demo/gt/gt.json)

  2. ensamblar.py  -> reconstruccion y visualizacion de las franjas (demo/salida/). 
                      Recibe solo el mosaico, R, C y el anclaje

  3. evaluar.py    -> A_pos y A_adj, leyendo el ground truth despues del ensamblado
Si todo funciona, al final aparece "A_pos": 1.0.
"""
import subprocess
import sys

img = sys.argv[1] if len(sys.argv) > 1 else 'imagenes/coffee.png'
R, C, d, sem = (sys.argv[2:6] + ['5', '7', '3', '0'][len(sys.argv[2:6]):])
py = sys.executable


def correr(args):
    print('\n$ ' + ' '.join(args[1:]))
    subprocess.run(args, check=True)


correr([py, 'randomizar.py', img, R, C, d, sem, '--salida', 'demo'])
correr([py, 'ensamblar.py', 'demo/mosaico.png', '--caso', 'demo/caso.json', '--salida', 'demo/salida', '--visualizar', 'armonico'])
correr([py, 'evaluar.py', 'demo/salida/resultado.json', 'demo/gt/gt.json'])
