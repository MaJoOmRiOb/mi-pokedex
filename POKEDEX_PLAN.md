# PokéDex IA — Plan de Trabajo

> Documento vivo. Se actualiza a medida que avanza el proyecto.

---

## ¿Qué es este proyecto?

Aplicación web tipo Pokédex con reconocimiento de imagen en tiempo real.
El usuario apunta la cámara a un Pokémon (físico, pantalla, dibujo) y la app identifica
cuál es y muestra todas sus estadísticas usando la PokéAPI.

**Alcance inicial:** solo Generación I (#001 – #151)

---

## Estructura del proyecto — dos partes independientes

Este proyecto se divide en dos bloques que se desarrollan en secuencia.
No se puede construir la app web hasta tener el modelo entrenado y exportado.

```
┌─────────────────────────────────────────────────────────────┐
│  PARTE 1 — Machine Learning                                 │
│                                                             │
│  Lenguaje: Python                                           │
│  Objetivo: entrenar un modelo CNN capaz de distinguir       │
│            los 151 Pokémon de la 1.ª generación             │
│  Output:   modelo exportado en formato TF.js                │
│            (archivos model.json + .bin)                     │
└──────────────────────────┬──────────────────────────────────┘
                           │  modelo listo
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  PARTE 2 — App Web                                          │
│                                                             │
│  Lenguaje: HTML / CSS / JavaScript vanilla                  │
│  Objetivo: interfaz tipo Pokédex que carga el modelo en     │
│            el navegador, accede a la cámara y consulta      │
│            la PokéAPI para mostrar los datos del Pokémon    │
│  Output:   web desplegada en GitHub Pages                   │
└─────────────────────────────────────────────────────────────┘
```

**Por qué este orden importa:** el modelo define el contrato (orden de las 151 clases)
que la app web debe respetar. Cambiar el modelo después obliga a actualizar la app.

---

## Idea general (flujo de la app)

```
[Cámara del usuario]
       │
       ▼
 Captura de frame
       │
       ▼
 Pre-procesamiento de imagen
 (resize 224×224, normalización)
       │
       ▼
 Modelo CNN en el navegador (TF.js)
 → outputs: [prob. bulbasaur, prob. ivysaur, ..., prob. mew] (151 clases)
       │
       ▼
 Clase con mayor probabilidad + score de confianza
       │
       ▼
 Llamada a PokéAPI   →   Datos del Pokémon
       │
       ▼
 Renderizado en la interfaz tipo Pokédex
```

---

## Tecnologías y herramientas

### Modelo de machine learning
| Herramienta | Rol |
|---|---|
| **Python 3.10+** | Lenguaje de entrenamiento |
| **TensorFlow / Keras** | Framework del modelo |
| **MobileNetV2** | Arquitectura base (transfer learning) |
| **scikit-learn** | Métricas, split de datos |
| **Matplotlib / Seaborn** | Gráficas de entrenamiento |
| **Pillow / OpenCV** | Pre-procesamiento de imágenes |
| **tensorflowjs_converter** | Exportar el modelo `.h5` → formato TF.js |

> **¿Por qué MobileNetV2?**
> Es ligera y precisa. Corre en el navegador con TF.js sin necesidad de servidor.
> El fine-tuning con ~150K imágenes converge en pocas épocas.

### App web
| Herramienta | Rol |
|---|---|
| **HTML5 / CSS3 / JS vanilla** | Sin frameworks — fácil de hostear en GitHub Pages |
| **TensorFlow.js** | Ejecutar el modelo entrenado directamente en el navegador |
| **PokéAPI** (`pokeapi.co`) | Stats, tipos, sprites, descripciones (REST, gratuita) |
| **Web Camera API** (`getUserMedia`) | Acceso a cámara del dispositivo |
| **Canvas API** | Captura de frames de video para inferencia |

> **¿Por qué PokéAPI?**
> Es la más completa para datos de Pokémon: stats base, tipos, habilidades, altura,
> peso, flavor text, sprites oficiales y shiny. Sin autenticación, sin límite agresivo.

### Hosting / Deploy
| Opción | Costo | Notas |
|---|---|---|
| **GitHub Pages** | Gratis | Ideal, el modelo vive en `/model/` como archivos estáticos |
| **Hugging Face Spaces** | Gratis | Alternativa si se quiere demo standalone |

---

## Dataset — la parte más importante

El modelo necesita imágenes de los 151 Pokémon para aprender a distinguirlos.
Hay tres fuentes realistas:

### Opción A — Dataset pre-armado de Kaggle ✅ (recomendado para empezar)
- **"Pokémon Generation One" en Kaggle** — tiene carpetas por Pokémon con imágenes
  de merchandise, cartas, renders 3D y ilustraciones
- Pros: listo para usar, bien etiquetado
- Cons: puede no tener suficiente variedad para imágenes "del mundo real"

### Opción B — Sprites oficiales + augmentation
- Descargar todos los sprites de la PokéAPI (múltiples vistas: front, back, shiny)
- Aplicar data augmentation agresivo: rotaciones, zoom, ruido, cambios de brillo
- Pros: consistente, fácil de obtener programáticamente
- Cons: los sprites son muy distintos a fotos reales de figuras o pantallas

### Opción C — Frames del anime (muy recomendado como complemento)
- Descargar episodios del anime original donde aparezca el Pokémon objetivo
- Extraer frames con `ffmpeg` y filtrar los más diversos entre sí
- Pros: imágenes del mundo real estilizado, múltiples ángulos, iluminación variada — exactamente lo que falta en datasets de sprites
- Cons: requiere un paso de cropping manual o automático para aislar al Pokémon del fondo

**Flujo de extracción de frames:**
```bash
# 1. Extraer 1 frame por segundo (suficiente para evitar duplicados obvios)
ffmpeg -i episodio.mp4 -vf fps=1 frames/frame_%04d.jpg

# O extraer solo keyframes (cambios de escena) — más eficiente
ffmpeg -i episodio.mp4 -vf "select=eq(pict_type\,I)" -vsync vfr frames/frame_%04d.jpg
```

Luego pasar esos frames por el pipeline de deduplicación de imágenes (ver Fase 1).

#### Herramienta de deduplicación de videos (para evitar procesar episodios repetidos)

Si se acumulan muchos archivos de video y se quiere detectar episodios duplicados o
versiones casi idénticas antes de extraer frames:

| Situación | Herramienta | Notas |
|---|---|---|
| Detectar videos duplicados o re-subidas | `videohash` (Python) | Compara por hash perceptual del video completo |
| Dataset grande de videos + imágenes | `fastdup` (v1.0+) | Soporta videos e imágenes en el mismo pipeline |
| Revisión manual de videos similares | VLC + comparación manual | Para colecciones pequeñas |

```bash
pip install videohash
```
```python
from videohash import VideoHash

hash1 = VideoHash(path="episodio_01.mp4")
hash2 = VideoHash(path="episodio_01_redub.mp4")
print(hash1 - hash2)  # distancia: 0 = idénticos, >10 = diferentes
```

> **Estrategia práctica:** no hace falta deduplicar los videos si se baja de fuentes
> confiables. El cuello de botella real está en los frames: un episodio de 22 min
> a 24fps genera ~31 000 frames. Extrayendo a 1fps ya son ~1320, y luego
> `imagededup` elimina los que son prácticamente iguales.

### Opción D — Web scraping de imágenes (avanzado)
- Scraping ético de Bulbapedia, Serebii, Pokéwalls
- Más variedad visual
- Cons: requiere limpieza manual, posible ruido

### ¿Y la IA generativa (Stable Diffusion, Midjourney)? — Leer antes de usarla

Técnicamente funciona: los modelos generativos conocen muy bien la 1.ª gen porque
son iconos globales sobrerepresentados en sus datos de entrenamiento.

**Pero para este proyecto no vale la pena como estrategia general**, por tres razones:

| Problema | Descripción |
|---|---|
| **Carga de revisión manual** | 100 img × 151 Pokémon = 15 100 imágenes que una sola persona debe revisar. Los generativos cometen errores sutiles: cola mal, colores incorrectos, rasgos mezclados entre especies. |
| **Ya existe una alternativa sin revisión** | Data augmentation clásico (rotaciones, flips, zoom, brillo, cutout) genera variedad sintética desde imágenes ya validadas, sin producir artefactos nuevos. Es lo que hace `train.py` de todas formas. |
| **Domain shift** | El modelo aprendería un "estilo generativo" que no corresponde ni a sprites, ni al anime, ni a fotos reales — puede empeorar la generalización en vez de mejorarla. |

**Cuándo SÍ tiene sentido usarla:** si tras reunir todas las fuentes algún Pokémon
concreto sigue con menos de 50 imágenes. En ese caso, generar 30-40 imágenes
solo para esa clase específica y revisarlas manualmente es razonable — el volumen
es manejable y el beneficio justifica el esfuerzo.

> **Estrategia recomendada:** Kaggle + sprites PokéAPI + frames del anime + data augmentation.
> IA generativa solo como último recurso para clases con muy pocas imágenes.

### Cantidad mínima de imágenes por Pokémon

El riesgo principal no es confundir Pokémon muy distintos, sino confundir Pokémon
de la misma línea evolutiva que comparten rasgos visuales.

**Casos de alto riesgo de confusión en Gen I:**

| Línea evolutiva | Riesgo | Por qué se confunden |
|---|---|---|
| **Grimer ↔ Muk** | **Crítico** | El mismo blob morado, Muk es solo más grande. La única diferencia es el tamaño y una cara ligeramente distinta |
| **Pidgey ↔ Pidgeotto ↔ Pidgeot** | **Crítico** | Aves marrones casi idénticas — difíciles incluso para humanos |
| Caterpie ↔ Metapod | Alto | Pequeños, verdes/blancos, forma similar |
| Ekans ↔ Arbok | Alto | Ambos serpientes moradas |
| Machop ↔ Machoke ↔ Machamp | Alto | Humanoides azul-grisáceo, mismos rasgos escalados |
| Geodude ↔ Graveler ↔ Golem | Alto | Rocas con cara, misma paleta |
| Gastly ↔ Haunter ↔ Gengar | Alto | Fantasmas púrpura/oscuro |
| Abra ↔ Kadabra ↔ Alakazam | Alto | Humanoides psíquicos, misma pose |
| Magikarp ↔ Gyarados | Bajo | Pez rojo vs dragón marino azul gigante |
| Caterpie ↔ Butterfree | Bajo | Oruga vs mariposa — cambio radical |

> **Regla práctica:** si tú mismo tardas más de 2 segundos en distinguirlos,
> el modelo va a necesitar el doble de imágenes. Si ya es difícil para un humano,
> el modelo solo tiene píxeles — no tiene contexto, no tiene escala, no tiene memoria.

**Mínimos recomendados por tipo de caso:**

| Caso | Mínimo real | Target ideal |
|---|---|---|
| Diseño único, sin evolutivas similares (Snorlax, Mewtwo, Ditto...) | **100 img** | 200–300 |
| Línea evolutiva con cambio drástico (Magikarp→Gyarados) | **150 img** | 300–400 |
| Línea evolutiva con diseño muy similar (Machop/Machoke/Machamp) | **250 img** | 400–500 |
| Casos críticos — difíciles incluso para humanos (Grimer/Muk, Pidgey line) | **350 img** | 500–700 |

**Mínimo global absoluto: 150 imágenes reales por Pokémon**, con augmentation 3×–4×
encima durante el entrenamiento, lo que da ~450–600 ejemplos efectivos por clase.

> El augmentation no reemplaza la variedad real en casos difíciles. Con 50 imágenes
> de Machop rotadas sigues teniendo 50 composiciones únicas del mismo ángulo.
> Para líneas evolutivas similares, necesitas variedad genuina de fuentes distintas.

---

## Estructura de carpetas del proyecto

```
pokedex-ia/
│
├── train/                      ← Todo lo del modelo Python
│   ├── data/
│   │   └── raw/                ← Imágenes descargadas (no se sube a Git)
│   │       ├── bulbasaur/
│   │       ├── ivysaur/
│   │       └── ...
│   ├── train.py                ← Script principal de entrenamiento
│   ├── download_sprites.py     ← Descarga sprites de PokéAPI automáticamente
│   ├── requirements.txt
│   └── notebooks/
│       └── exploration.ipynb   ← EDA y experimentos
│
├── model/                      ← Modelo exportado para TF.js (se sube a Git)
│   ├── model.json
│   └── group1-shard1of1.bin
│
├── app/                        ← App web
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── pokemon_names.js        ← Array con los 151 nombres (orden = clase del modelo)
│
└── README.md
```

---

## Roadmap

### Fase 1 — Datos (1–2 días)
- [ ] Buscar y descargar dataset de Kaggle
- [ ] Complementar con búsqueda de imágenes en Google / buscadores por Pokémon
- [ ] Escribir `download_sprites.py` para complementar con sprites de la PokéAPI
- [ ] **Deduplicación** — eliminar imágenes iguales o muy similares entre las fuentes
- [ ] Revisar calidad: que las 151 carpetas existan y tengan al menos 100 imágenes cada una
- [ ] **Definir y fijar el orden de las 151 clases** — este orden es el contrato
      entre el modelo y la app web (ej. índice 0 = bulbasaur, índice 1 = ivysaur...)

#### Herramienta de deduplicación según volumen de imágenes por Pokémon

| Imágenes por clase | Herramienta recomendada | Método | Notas |
|---|---|---|---|
| < 200 img | `imagededup` (CNN) | Hash neuronal | Detecta duplicados aunque cambien resolución, formato o recorte |
| 200 – 1 000 img | `imagededup` (CNN) | Hash neuronal | Sigue siendo la mejor opción, sigue siendo rápida |
| > 1 000 img | `fastdup` | Embeddings + índice ANN | Mucho más rápido a escala; también detecta imágenes borrosas o corruptas |
| Revisión manual | dupeGuru (GUI) | Comparación visual | Útil para verificar casos dudosos que los scripts marquen como similares |

**Instalación rápida:**
```bash
pip install imagededup   # para volúmenes pequeños/medianos
pip install fastdup      # para volúmenes grandes
```

**Uso típico con `imagededup` (por carpeta/clase):**
```python
from imagededup.methods import CNN

dedup = CNN()
duplicates = dedup.find_duplicates(image_dir='data/raw/pikachu/', max_distance_threshold=10)
dedup.plot_duplicates(duplicates, image_dir='data/raw/pikachu/')  # mapa visual
```

> Correr esto **por carpeta** (una por Pokémon), no sobre todo el dataset junto,
> para no confundir imágenes similares entre especies distintas (ej. Caterpie vs Metapod).

### Fase 2 — Entrenamiento (2–3 días)
- [ ] Escribir `train.py` con MobileNetV2 + fine-tuning
- [ ] Splits: 80% train / 10% val / 10% test
- [ ] Data augmentation (rotación ±30°, flip, zoom, brillo)
- [ ] Entrenar con GPU (Google Colab gratis es suficiente para este tamaño)
- [ ] Métricas objetivo: **accuracy > 85%** en test
- [ ] Guardar modelo en `.h5`
- [ ] Convertir a TF.js con `tensorflowjs_converter`

### Fase 3 — App web (2–3 días)
- [ ] Interfaz Pokédex: cámara en vivo + panel de resultado
- [ ] Cargar modelo TF.js y hacer inferencia sobre el frame capturado
- [ ] Llamada a PokéAPI para obtener: stats, tipos, altura, peso, habilidades, sprite
- [ ] Mostrar top-3 predicciones con porcentaje de confianza
- [ ] Responsive (funciona bien en móvil — la cámara del celular es el caso de uso principal)

### Fase 4 — Deploy y presentación (1 día)
- [ ] Subir a GitHub con README explicativo
- [ ] Deploy en GitHub Pages
- [ ] Demo GIF para el README y para el portafolio

---

## Puntos críticos a tener en cuenta

### El modelo no va a reconocer fotos reales perfectamente al principio
Los datasets de Pokémon suelen tener ilustraciones/sprites, no fotos de figuras del
mundo real. Si se quiere que funcione con fotos reales de juguetes o pantallas,
hay que asegurarse de incluir esa variedad en el dataset.

### El orden de las clases es crítico
El modelo asigna probabilidades a índices numéricos (0, 1, 2...).
`pokemon_names.js` debe tener exactamente el mismo orden que las carpetas de
entrenamiento ordenadas alfabéticamente (que es como Keras las lee por defecto).
Un error aquí hace que la app identifique Pikachu como Mewtwo.

### Confianza baja = no mostrar resultado
Poner un umbral (ej. 60%). Si el modelo no está seguro, mejor decirlo que mostrar
datos incorrectos.

### CORS con TF.js en localhost
Al probarlo localmente, el modelo debe servirse desde un servidor HTTP, no abriendo
el HTML como archivo. Usar `npx serve` o la extensión Live Server de VSCode.

---

## Recursos clave

| Recurso | URL |
|---|---|
| PokéAPI docs | `https://pokeapi.co/docs/v2` |
| Dataset Kaggle (ejemplo) | `https://www.kaggle.com/datasets/lantian773030/pokemonclassification` |
| TF.js guía de conversión | `https://www.tensorflow.org/js/guide/conversion` |
| MobileNetV2 paper | Transfer learning con ~3.4M parámetros, ideal para móvil |
| TF.js modelo layers | `tf.loadLayersModel('model/model.json')` |

---

## Resumen de prioridades

```
1. Conseguir y verificar el dataset         ← Sin esto no hay modelo
2. Definir el orden de las 151 clases       ← Contrato modelo ↔ app
3. Entrenar con Colab (gratis, rápido)      ← Fase más técnica
4. Convertir y probar en navegador          ← El punto de integración
5. Construir la UI                          ← Lo más visible, pero lo último
```
