# HerdNet CBAM - Detección de Animales en Imágenes Aéreas

## Integrantes del Equipo
- Victor Pérez
- Jordi Sanchez
- Maryi Carvajal
- Simón Aristizábal

Universidad de los Andes - Maestría en Inteligencia Artificial (MAIA)

## Descripción del Proyecto
Este proyecto implementa una extensión del modelo HerdNet mediante la incorporación de un módulo de atención CBAM (Convolutional Block Attention Module) para mejorar la detección y conteo de animales en manadas densas a partir de imágenes aéreas. El CBAM permite al modelo enfocar su atención tanto en regiones espaciales como en canales relevantes del cuello de botella después del backbone del modelo principal.

## Estructura del Proyecto
```
maia_proyecto_final/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes/
│   │   │       ├── inference.py
│   │   │       └── ping.py
│   │   ├── core/
│   │   │   └── config.py
│   │   ├── herdnet/
│   │   │   ├── full_model.pth
│   │   │   ├── infer-custom.py
│   │   │   ├── data/
│   │   │   ├── animaloc/
│   │   ├── images/
│   │   ├── main.py
│   │   └── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .frontend-env/
├── notebooks/
│   ├── train_test_herdnet_dla60.ipynb 
│   ├── train_herdnet_cbam.ipynb
│   └── test_herdnet_cbam.ipynb
├── datos/
│   ├── sample_00_cropped.JPG
│   ├── sample_01.JPG
│   ├── sample_02.JPG
│   └── sample_03.JPG
├── documentos/
│   └── Informe_final.pdf
├── modelos/ 
│   └── herdnet_cbam_full_model.pth
├── docker-compose.yml
└── README.md
```

En esta estructura hacen falta los modelos de DLA60 entrenados, los cuales tienen un tamaño superior al permitido por Github. Por lo cual fueron alojados en [Google Drive](https://drive.google.com/drive/folders/1-XwV-6Sp7P04gw3d7YDeeQgVUXqTzRTt?usp=sharing).

## Requisitos
- Docker
- Docker Compose
- Python 3.8
- Git

## Configuración del Entorno

### 1. Clonar el Repositorio
```bash
git clone https://github.com/victordpd19/maia_proyecto_final.git
cd maia_proyecto_final
```

### 2. Configuración del Backend
El backend está construido con FastAPI y contiene el modelo HerdNet CBAM modificado. Los archivos principales son:
- `backend/app/main.py`: Punto de entrada de la API
- `backend/app/api/routes/inference.py`: Endpoints para inferencia
- `backend/app/herdnet/infer-custom.py`: Script de inferencia personalizado

### 3. Configuración del Frontend
El frontend está construido con Streamlit y proporciona una interfaz web para:
- Carga de imágenes
- Visualización de resultados
- Filtrado de detecciones por umbral de confianza
- Visualización de puntos de detección

## Ejecución del Proyecto

### Usando Docker Compose
```bash
docker-compose up --build
```

Esto iniciará:
- Backend en http://localhost:8000
- Frontend en http://localhost:8501

### Variables de Entorno
Crear un archivo `.env` en la raíz del proyecto:
```
API_URL=http://localhost:8000
```

## Uso de la Aplicación
1. Acceder a http://localhost:8501
2. Subir una imagen (formatos soportados: jpg, jpeg, png)
3. Hacer clic en "Process Image"
4. Ajustar el umbral de confianza según sea necesario
5. Visualizar los resultados:
   - Imagen procesada
   - Puntos de detección
   - Datos de detección

## Advertencias y Consideraciones
⚠️ **Importante: Rendimiento del Procesamiento**

Debido a restricciones de hardware en los servidores de despliegue, el modelo se ejecuta en CPU en lugar de GPU. Esto implica:

- El procesamiento de imágenes de alta resolución (4000x6000 píxeles) puede tomar entre 2 a 5 minutos
- El tiempo de procesamiento varía según el tamaño de la imagen
- Se recomienda paciencia durante el proceso de inferencia :)


Para un rendimiento óptimo en entornos locales con GPU disponible, se puede modificar el Dockerfile del backend para utilizar una imagen base de CUDA*.

* Si se desea usar GPU, es necesario modificar el valor de **DEVICE** en el archivo .env por el valor de "cuda", además de instalar la correcta version en el dockerfile para uso de GPUs.


## Características del Modelo
- Arquitectura: Red Neuronal Convolucional con CBAM (Convolutional Block Attention Module)
- Tamaño de entrada: Cualquier tamaño de imagen (Stitcher enabled)
- Tamaño de ventana deslizante: 512x512 píxeles
- Overlap: 160 píxeles
- Clases de salida: 6 especies animales (kob, topi, waterbuck, elephant, buffalo, warthog)

## Métricas de Rendimiento

| Métrica    | Valor  |
|------------|--------|
| MAE        | 3.791  |
| MSE        | 62.884 |
| RMSE       | 7.930  |
| Accuracy   | 0.932  |
| F1 Score   | 0.704  |
| mAP        | 0.565  |
| Precision  | 0.634  |
| Recall     | 0.792  |

## Casos de Uso
- Monitoreo de poblaciones de vida silvestre
- Detección de animales en hábitats naturales
- Investigación en conservación
- Estudios ecológicos

## Dataset
El modelo fue entrenado utilizando el dataset de detección e identificación de mamíferos africanos en imágenes aéreas:
[Dataset Link](https://dataverse.uliege.be/dataset.xhtml?persistentId=doi:10.58119/ULG/MIRUU5)
