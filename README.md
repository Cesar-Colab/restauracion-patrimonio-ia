# Aplicación IA - Restauración y Levantamiento de Fachadas

Esta aplicación permite restaurar imágenes históricas de fachadas patrimoniales y estimar su profundidad usando inteligencia artificial (MiDaS + Streamlit).

## Cómo usar
1. Sube una imagen JPG o PNG antigua.
2. La app simula una restauración visual.
3. Se genera y visualiza el mapa de profundidad 2D.

## Requisitos
- Python 3.9+
- GPU (opcional)

## Ejecución local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Estructura
- `app.py` → Lógica principal de la app.
- `models/` → Contiene modelo preentrenado MiDaS.
- `requirements.txt` → Dependencias.
