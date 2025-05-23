# HerdNet CBAM - Detección de Animales en Imágenes Aéreas

## Integrantes del Equipo
- Victor Pérez
- Jordi Sanchez
- Maryi Carvajal
- Simón Aristizábal

**Universidad de los Andes** - **Maestría en Inteligencia Artificial (MAIA)**

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
│   └── Dockerfile
├── notebooks/ # notebooks para ejecutar tests sobre herdnet cbam y train/test para dla60
│   ├── train_test_herdnet_dla60.ipynb
│   └── test_herdnet_cbam.ipynb
├── datos/ # estas imagenes de pruebas al momento de ejecutar la aplicacion para hacer la inferencia.
│   ├── sample_00_cropped.JPG
│   ├── sample_01.JPG
│   ├── sample_02.JPG
│   └── sample_03.JPG
├── documentos/ # última iteración del informe entregado
│   └── Informe_final.pdf
├── modelos/ # modelo de cbam comprimido a ~70MB, los otros modelos se encuentren en el drive mencionado más adelante
│   └── herdnet_cbam_full_model.pth
├── docker-compose.yml
└── README.md
```

En esta estructura hacen falta los modelos de DLA60 y base entrenados dentro de la carpeta `modelos/`, los cuales tienen un tamaño superior al permitido por GitHub. Por lo cual fueron alojados en [Google Drive](https://drive.google.com/drive/folders/1-XwV-6Sp7P04gw3d7YDeeQgVUXqTzRTt?usp=sharing).

## Ejecución de notebooks en colab:
Para ejecutar los notebooks de este repositorio en colab, puede seguir los siguientes links. En el caso de CBAM, este fue entrenado en una VM independiente utilizando la version modificada del repositorio original de Herdnet, encontrado en [HerdnetCBAM](https://github.com/victordpd19/HerdNet).

  
* Test-only: **Herdnet CBAM:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/victordpd19/maia_proyecto_final/blob/main/notebooks/test_herdnet_cbam.ipynb)
  
* Train-test: **Herdnet Base:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/victordpd19/maia_proyecto_final/blob/main/notebooks/train_test_hernet_base.ipynb)
  
* Train-test: **DLA60:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/victordpd19/maia_proyecto_final/blob/main/notebooks/train_test_herdnet_dla60.ipynb)


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

El proyecto está desplegado en una instancia de GCP tipo n1-standard-4 (4 vCPUs, 15 GB Memory) con GPU (P4). Para dicha instancia, se siguieron algunos pasos de instalación de controladores de GPU previo a poder ejecutar la aplicación desde el docker compose:

* Instalación de docker y docker compose.
* Instalación de los drivers de nvidia-driver-570.
* Instalación del container-toolkit.

Para ejecutar el proyecto en la instancia, se debe estar sobre la carpeta clonada del repositorio, sobre la rama main, y seguir los pasos a continuación.

### Usando Docker Compose
```bash
docker-compose up --build
```

Esto iniciará:
- Backend en http://localhost:8000
- Frontend en http://localhost:8501

Cabe resaltar, que la instancia fue enlazada con una IP estática: (http://35.208.185.101:8501/). En esta dirección, se puede acceder para realizar inferencias sobre imágenes aereas, como los ejemplos disponibles en la carpeta **datos/** de este repositorio.

### Variables de Entorno
Crear un archivo `.env` en la raíz del proyecto:
```
API_URL=http://localhost:8000
```
Esto es para que el servicio del front, pueda descubrir el servicio del backend, asumiendo que ambos están en la misma subred y sus puertos son los mencionados anteriormente (8000 y 8501).


## Uso de la Aplicación
1. Acceder a http://35.208.185.101:8501/
2. Subir una imagen (formatos soportados: jpg, jpeg)
3. Hacer clic en "**Process Image**"
4. Ajustar el umbral de confianza según sea necesario.
5. Visualizar los resultados:
   - Imagen procesada
   - Puntos de detección
   - Datos de detección

## Características del Modelo
- Arquitectura: Red Neuronal Convolucional con CBAM (Convolutional Block Attention Module)
- Tamaño de entrada: Cualquier tamaño de imagen (Stitcher enabled)
- Tamaño de ventana deslizante: 512x512 píxeles
- Overlap: 160 píxeles
- Clases de salida: 6 especies animales (kob, topi, waterbuck, elephant, buffalo, warthog)

## Métricas de Rendimiento


| Métrica | MAE | MSE | RMSE | Accuracy | F1 Score | mAP | Precision | Recall |
|---------|-----|-----|------|----------|----------|-----|-----------|---------|
| **Valor** | 3.791 | 62.884 | 7.930 | 0.932 | 0.704 | 0.565 | 0.634 | 0.792 |


## Casos de Uso
- Monitoreo de poblaciones de vida silvestre
- Detección de animales en hábitats naturales
- Investigación en conservación
- Estudios ecológicos

## Dataset
El modelo fue entrenado utilizando el dataset de detección e identificación de mamíferos africanos en imágenes aéreas: [Dataset Link](https://dataverse.uliege.be/dataset.xhtml?persistentId=doi:10.58119/ULG/MIRUU5). 




