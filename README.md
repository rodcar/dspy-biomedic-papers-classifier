# Clasificador Biomédico DSPy 

Clasificador multi-etiqueta de artículos biomédicos usando DSPy y modelos de lenguaje optimizados.

## 🎯 Descripción

Sistema de clasificación automática que categoriza papers biomédicos en 4 sistemas orgánicos:
- **Neurológico**: Sistema nervioso, cerebro, trastornos neurológicos
- **Cardiovascular**: Corazón, vasos sanguíneos, hipertensión
- **Hepatorenal**: Hígado y riñones, función renal/hepática  
- **Oncológico**: Cáncer, tumores, neoplasias

## 🚀 Instalación y Uso

### Prerrequisitos
- Python 3.12+
- Clave API de OpenAI

### Configuración
```bash
# Clonar repositorio
git clone https://github.com/rodcar/dspy-biomedic-papers-classifier.git
cd dspy-biomedic-papers-classifier

# Instalar uv (si no lo tienes)
pip install uv

# Crear entorno virtual
uv venv

# Instalar dependencias
uv sync

# Configurar API key en .env
echo "OPENAI_API_KEY=tu_clave_api" > .env
```

### Ejecución
```bash
source .venv/bin/activate
# Evaluar modelo con archivo CSV
# Reemplaza directament `data/input.csv` o use otra ruta.
# Este archivo tiene pocos ejemplos para comprobar la funcionalidad.
uv run python main.py data/input.csv

# El CSV debe tener columnas: title;abstract;group
```

## 📊 Resultados

Ver `notebook/notebook_gepa.ipynb`.

**Métricas obtenidas en dataset de prueba (1,213 ejemplos):**
- **F1 Score Promedio**: 0.8216
- F1 Neurológico: 0.7612 (76.1%)
- F1 Cardiovascular: 0.8164 (81.6%)  
- F1 Hepatorenal: 0.8340 (83.4%)
- F1 Oncológico: 0.8748 (87.5%)
- **Métrica DSPy**: 84.4%

## 🏗️ Arquitectura

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CSV Input     │───▶│  DSPy Classifier │───▶│  Predictions +  │
│ (title,abstract)│    │   (GEPA Opt.)    │    │    Metrics     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                      ┌──────────────────┐
                      │ OpenAI GPT-4o    │
                      │   (Backend)      │
                      └──────────────────┘
```

## 📁 Estructura

```
├── main.py                    # Script principal de evaluación
├── programs/                  # Modelos entrenados DSPy
│   └── multilabel_classifier_gepa_optimized.json
├── data/                      # Datasets
│   ├── input.csv             # Datos de prueba
│   └── challenge_data-18-ago.csv
├── notebook/
│   └── notebook_gepa.ipynb   # Desarrollo y optimización
└── README.md                 # Este archivo
```

## 🔬 Metodología

1. **Preprocesamiento**: Carga de datos CSV con formato específico
2. **Clasificación**: Modelo DSPy optimizado con GEPA
3. **Evaluación**: F1 score ponderado y matrices de confusión
4. **Salida**: CSV con predicciones + métricas visuales

## 📈 Salida

El sistema genera:
- CSV con columna `group_predicted`
- F1 score ponderado (métrica principal)
- Matrices de confusión por categoría
- Gráficos de evaluación

## 🛠️ Desarrollo

Modelo optimizado usando:
- **DSPy Framework**: Optimización automática de prompts
- **GEPA Optimizer**: Algoritmo genético para mejora
- **Chain-of-Thought**: Razonamiento paso a paso
- **Multi-label**: Clasificación en múltiples categorías

---
**Autor**: Ivan Rodriguez  
**Framework**: DSPy + OpenAI GPT-4o-mini
