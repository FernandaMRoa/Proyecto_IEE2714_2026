# Proyecto Semestral IEE2714 (2° 2026) — Reconstrucción automática de mosaicos de parches

Pontificia Universidad Católica de Chile — Fundamentos de Procesamiento de Imágenes

Integrantes: José Gerónimo Nehuenmapu Alarcón Lipin · María Fernanda Muñoz Roa

Toolbox en Python que reconstruye automáticamente una imagen RGB dividida en una grilla de
R×C parches permutados y separados por franjas de ceros de ancho desconocido
d ∈ {0, 1, 3, 5}. El ensamblador recibe solo la imagen, R, C y un anclaje. El interleave d se
estima automáticamente y no hay parámetros que ajustar por imagen.

> **Estado del repositorio.** Ya está todo el software: randomizador, estimación del
> interleave, compatibilidad, ensamblador (con y sin anclaje), visualización, evaluación y demo.
> Faltan los **scripts de experimentos** para generar tablas y figuras en el informe.

---

## 1. Instalación y prueba rápida

Requiere Python ≥ 3.10 (probado con 3.14).

```bash
git clone https://github.com/FernandaMRoa/Proyecto_IEE2714_2026.git
cd Proyecto_IEE2714_2026
pip install -r requirements.txt
python demo.py
```

`demo.py` desordena `imagenes/coffee.png`, la reconstruye y la evalúa. Si todo está bien, al
final aparece `"A_pos": 1.0`.

Dependencias: `numpy`, `scipy`, `scikit-image` y `matplotlib`. No se usa machine learning,
descriptores externos ni solvers externos de ensamblaje u optimización: el descriptor, la
función de compatibilidad y el algoritmo de ensamblaje están implementados en `mosaico/`.

**Todos los comandos se ejecutan desde la carpeta raíz del repositorio.**

---

## 2. Estructura

```
├── ensamblar.py          PUNTO DE ENTRADA PRINCIPAL: reconstruye un mosaico
├── randomizar.py         genera casos de prueba (mosaico + ground truth)
├── evaluar.py            evaluación con ground truth (A_pos, A_adj)
├── visualizar.py         rellena las franjas para ver la imagen continua
├── demo.py               flujo completo de ejemplo
├── requirements.txt
├── imagenes/             imágenes de ejemplo (coffee.png, rocket.png)
└── mosaico/              paquete con toda la lógica
    ├── geometria.py      convención de crop/interleave, validación de rangos, índices
    ├── imagenes.py       lectura/escritura como RGB uint8
    ├── randomizador.py   cortar, armar y desordenar mosaicos, ground truth
    ├── evaluacion.py     métricas A_pos y A_adj
    ├── interleave.py     estimación automática de d y extracción de parches
    ├── compatibilidad.py descriptores de borde y medidas SSD, predicción y MGC
    ├── visualizacion.py  relleno de franjas: lineal/bilineal, Hermite cúbico, Laplace
    └── ensamblador.py    normalización, armado voraz, búsqueda local, segmentos, bonus
```

---

## 3. Flujo de uso completo

```bash
# 1. Crear un caso: imagen, R, C, d, semilla
python randomizar.py imagenes/coffee.png 5 7 3 0 --salida caso

# 2. Reconstruirlo: solo mosaico, R, C y anclaje (el anclaje lo imprime el paso 1)
python ensamblar.py caso/mosaico.png 5 7 4 6 3 4 --salida salida --visualizar armonico

# 3. Evaluarlo con el ground truth (solo después de ensamblar)
python evaluar.py salida/resultado.json caso/gt/gt.json
```

Las carpetas `caso/`, `salida/` y `demo/` están en `.gitignore`: se pueden usar libremente
para pruebas.

---

## 4. Ensamblador (`ensamblar.py`)

### Entradas obligatorias

```bash
python ensamblar.py MOSAICO R C RS CS RT CT --salida CARPETA
```

| Entrada | Descripción |
|---|---|
| `MOSAICO` | imagen desordenada, RGB de 8 bits. Usar PNG o TIF (con JPEG las franjas dejan de ser cero exacto) |
| `R C` | filas y columnas de la grilla |
| `RS CS RT CT` | anclaje: la pieza de la celda (RS, CS) del mosaico va en la celda (RT, CT) de la imagen correcta |

**Convención de índices:** en la línea de comandos y en todos los archivos se usan índices
**1-based** (filas 1..R, columnas 1..C), igual que el enunciado. Internamente el código usa
0-based; la conversión está en `mosaico/geometria.py`.

### Opciones

| Opción | Por defecto | Descripción |
|---|---|---|
| `--salida CARPETA` | `salida` | carpeta de resultados |
| `--caso caso.json` | — | toma R, C y el anclaje del archivo público del randomizador |
| `--sin-anclaje` | — | **modo bonus**: reconstruye sin usar el anclaje |
| `--visualizar METODO` | — | guarda también la imagen con las franjas rellenadas (`lineal`, `cubico`, `armonico`) |
| `--config JSON` | ver tabla | cambia la configuración por defecto del ensamblador |

### Salidas (en `CARPETA/`)

| Archivo | Contenido |
|---|---|
| `reconstruida.png` | imagen reordenada, mismo tamaño que la entrada (franjas en cero) |
| `P_estimada.txt` | matriz P de R×C: en cada celda, el índice k = (r_s − 1)·C + c_s de la pieza del mosaico que se puso ahí |
| `resultado.json` | P, d estimado, tiempos por etapa (s) y diagnóstico: puntaje de cada candidato de d, modo de estimación, energía y solución elegida, configuración usada |
| `visualizacion.png` | solo con `--visualizar` |

### Configuración (`--config`)

Valores por defecto (`CONFIG_DEFECTO` en `mosaico/ensamblador.py`). El benchmark se corre
siempre con estos valores; `--config` es solo para experimentar.

| Clave | Defecto | Significado |
|---|---|---|
| `metodo` | `"mgc+pred"` | medida: `mgc`, `pred`, `ssd` o `mgc+pred` |
| `espacio` | `"lab"` | espacio de color de los bordes: `lab`, `rgb`, `gris` |
| `peso_pred` | `0.5` | peso del término de predicción en `mgc+pred` |
| `normalizacion` | `"segundo"` | `segundo` (razón a la 2ª mejor alternativa), `mediana`, `ninguna` |
| `tau` | `0.5` | suavizado de la normalización (fracción de la mediana de las mejores uniones) |
| `suavizado_aditivo` | `true` | usa (D+τ)/(D₂+τ): las piezas casi idénticas quedan neutras |
| `semillas_extra` | `6` | soluciones adicionales desde otras semillas (marco flotante) |
| `refinar` | `true` | búsqueda local (intercambios y desplazamientos de bloques) |
| `segmentos` | `true` | re-ensamblado por segmentos (vecindad grande) |
| `max_iter_refinamiento` | `50` | máximo de rondas de la búsqueda local |

Ejemplo (sin etapas de mejora):

```bash
# Linux, Mac, Git Bash o PowerShell 7
python ensamblar.py caso/mosaico.png --caso caso/caso.json --config '{"refinar": false, "segmentos": false, "semillas_extra": 0}'

# Windows PowerShell 5.1: las comillas internas se escapan con \"
python ensamblar.py caso/mosaico.png --caso caso/caso.json --config '{\"refinar\": false, \"segmentos\": false, \"semillas_extra\": 0}'
```

### Uso desde Python

```python
from mosaico.imagenes import leer_rgb
from mosaico.ensamblador import ensamblar

res = ensamblar(leer_rgb('caso/mosaico.png'), R=5, C=7, anclaje=(4, 6, 3, 4))
res['imagen'], res['P'], res['d'], res['tiempos'], res['diagnostico']

res_bonus = ensamblar(leer_rgb('caso/mosaico.png'), 5, 7, anclaje=None)   # sin anclaje
```

---

## 5. Randomizador (`randomizar.py`)

```bash
python randomizar.py IMAGEN R C D SEMILLA --salida CARPETA
```

Hace el crop centrado de R·⌊H₀/R⌋ × C·⌊W₀/C⌋, corta los R·C parches útiles de
(⌊H₀/R⌋ − d) × (⌊W₀/C⌋ − d), los pone en celdas de ceros (q = ⌊d/2⌋ ceros antes del parche y
q + 1 después), los desordena y sortea un anclaje. Con la misma semilla se obtiene exactamente
el mismo caso. Rechaza las configuraciones fuera del rango oficial (R o C fuera de 4..9,
parches fuera de 64..512 px, d ∉ {0, 1, 3, 5}); `--no-estricto` desactiva esa validación para
pruebas.

| Archivo | Contenido | ¿Lo ve el ensamblador? |
|---|---|---|
| `mosaico.png` | mosaico desordenado | sí |
| `caso.json` | R, C y anclaje (1-based) | sí |
| `gt/gt.json` | P_GT, d, semilla, geometría y crop | **no** (solo `evaluar.py`) |
| `gt/referencia.png` | imagen correcta con franjas en cero | no |

`P_GT` usa la misma convención que la salida P del ensamblador: `P_GT[r, c]` es el índice k de
la pieza del mosaico que va en la celda (r, c) de la imagen correcta. Así A_pos se calcula
comparando directamente las dos matrices.

---

## 6. Evaluación (`evaluar.py`)

```bash
python evaluar.py RESULTADO GT
python evaluar.py salida/resultado.json caso/gt/gt.json
```

`RESULTADO` es `resultado.json` (o `P_estimada.txt`) y `GT` es `gt/gt.json`. Calcula:

- **A_pos**: fracción de piezas en su celda correcta.
- **A_adj**: fracción de las R(C−1) + (R−1)C adyacencias verdaderas (izquierda→derecha y
  arriba→abajo) presentes en la reconstrucción, aunque el par esté desplazado.
- **perfecta**: si A_pos = 1; y, con `resultado.json`, si d se estimó bien.

Muestra las métricas y las guarda en `evaluacion.json`, junto a `RESULTADO`.

---

## 7. Visualización (`visualizar.py`)

```bash
python visualizar.py RECONSTRUIDA R C [--d D] [--metodo METODO] [--salida ARCHIVO]
python visualizar.py salida/reconstruida.png 5 7 --metodo armonico --salida salida/vis.png
```

Rellena las franjas solo para visualizar (la evaluación es sobre el orden de los parches). Si no
se da `--d`, se estima igual que en el ensamblador.

| Método | Descripción |
|---|---|
| `lineal` (defecto) | lineal entre bordes en las franjas y bilineal en los cruces (mínimo pedido) |
| `cubico` | Hermite cúbico, transición continua en pendiente; el mejor con d = 1 |
| `armonico` | ecuación de Laplace en las franjas; el mejor con d ≥ 3 |

El borde externo de la imagen se rellena repitiendo el valor del borde.

---

## 8. Demo (`demo.py`)

```bash
python demo.py                                  # coffee.png, 5x7, d = 3, semilla 0
python demo.py imagenes/rocket.png 5 6 5 3      # IMAGEN R C D SEMILLA
```

Ejecuta randomizar → ensamblar (con visualización armónica) → evaluar y guarda todo en `demo/`.
Sirve como prueba rápida después de instalar. En la demostración en vivo se usa directamente
`ensamblar.py` con el mosaico entregado.

---


## 10. Resumen del método

1. **Interleave.** Con R y C conocidos, cada candidato d predice exactamente qué filas y
   columnas locales de cada celda son separación. Se elige el mayor d cuyas franjas son cero en
   *todas* las celdas y en ambas direcciones (los candidatos están anidados). El contenido negro
   de la imagen no engaña al criterio porque, tras desordenar, no aparece en la misma fila local
   de todas las piezas. Si ningún d > 0 es exactamente consistente (por ejemplo, un mosaico en
   JPEG), un modo tolerante usa el perfil de brillo medio por fila local.
2. **Compatibilidad.** Solo se comparan los bordes, en CIE L\*a\*b\*. Se combina (i) la
   distancia de Mahalanobis del gradiente (MGC): el salto que cruza la unión, dividido por
   d + 1, se compara con la media y covarianza 3×3 del gradiente del borde, y (ii) una
   predicción lineal a través de la unión con norma L_p, p = 0,3. Los costos se normalizan
   respecto de la segunda mejor alternativa de ambas piezas.
3. **Ensamblaje.** Se minimiza la energía (suma de costos de todas las uniones): armado voraz
   (greedy) por confianza desde el anclaje, más soluciones de marco flotante desde las piezas
   con más "mejores amigos"; búsqueda local (intercambios y desplazamientos cíclicos de bloques)
   y re-ensamblado por segmentos. Se elige la solución de menor energía.
4. **Sin anclaje (bonus).** Las mismas etapas, solo con marco flotante y sin celda fija: la
   posición absoluta queda determinada porque la caja final debe medir exactamente R×C.

---

## 11. Limitaciones conocidas

- Texturas muy repetitivas (fachadas, rejas), simetrías y regiones uniformes generan piezas
  casi intercambiables; ahí la medida puede preferir un orden incorrecto.
- Si **todas** las piezas tuvieran su primera y última fila/columna exactamente en cero, el
  mosaico sería idéntico píxel a píxel al de un d mayor (caso no identificable). El estimador
  devuelve el mayor d consistente.
- Con mosaicos guardados en JPEG el modo tolerante estima bien d, pero el ensamblaje pierde
  exactitud porque la compresión contamina los bordes. Se recomienda PNG o TIF.

---