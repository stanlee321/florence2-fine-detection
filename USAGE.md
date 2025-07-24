# Florence-2 Fine Detection - Guía de Uso

## Mejoras de Rendimiento

### 1. Actualización Incremental de complete.json
El sistema ahora actualiza el archivo `complete.json` en MinIO después de procesar cada imagen:
- El archivo se sobrescribe con los resultados acumulados
- Incluye un campo `status` que indica si está "processing" o "complete"
- Permite consumir resultados parciales del mismo archivo
- No necesitas esperar a que termine todo el procesamiento

### 2. Envío Incremental de Notificaciones
Cada actualización envía una notificación a Kafka con la ruta del `complete.json` actualizado

### 2. Modos de Procesamiento

#### FULL (por defecto)
Procesa todas las imágenes y todos los objetos detectados en cada imagen.
```bash
python main.py
```

#### FAST
Omite el procesamiento de crops individuales, solo procesa imágenes completas.
```bash
PROCESS_MODE=FAST python main.py
```

#### CROPS_ONLY
Procesa SOLO los crops detectados, omite la descripción de la imagen completa.
```bash
CROPS_ONLY=true python main.py
```

### 3. Variables de Configuración

```bash
# Modo debug - muestra más información
DEBUG_MODE=true python main.py

# Limitar número de timestamps a procesar
MAX_TIMESTAMPS=2 python main.py

# Modo de procesamiento (FULL/FAST)
PROCESS_MODE=FAST python main.py

# Procesar SOLO crops (omite imagen completa)
CROPS_ONLY=true python main.py

# Procesamiento en paralelo (default: true)
PARALLEL_PROCESSING=true python main.py

# Número de workers para procesamiento paralelo (0 = 80% de CPUs)
MAX_WORKERS=0 python main.py

# Limitar número de crops por imagen (0 = todos)
MAX_CROPS_PER_IMAGE=5 python main.py

# Factor de escala para crops (default 8.0)
SCALE_FACTOR=4.0 python main.py

# Combinación óptima para procesamiento rápido de crops
CROPS_ONLY=true PARALLEL_PROCESSING=true MAX_WORKERS=0 python main.py

# Procesamiento rápido con límites
CROPS_ONLY=true MAX_CROPS_PER_IMAGE=10 SCALE_FACTOR=4.0 python main.py

# Debug con procesamiento paralelo
DEBUG_MODE=true MAX_TIMESTAMPS=1 CROPS_ONLY=true python main.py
```

## Estructura de Mensajes Kafka

### Mensaje Parcial (por cada imagen procesada)
```json
{
    "timestamp": "2024-01-01T12:00:00",
    "item_index": 0,
    "total_items": 26,
    "partial": true,
    "complete_json_path": "my-bucket/video-id/job-id/fine_detections/complete.json",
    "full_minio_path": "my-bucket/video-id/job-id/fine_detections/complete.json"
}
```

### Estructura del complete.json (actualizado incrementalmente)
```json
{
    "video_id": "01983d7d-06a5-76c1-b98c-970cd8bc9362",
    "job_id": "01983d74-dfe5-732b-b2ea-d0d5cca2fcb0",
    "model_id": "0194d257-9f70-7dfa-b56d-5fd1efb6365d",
    "total_timestamps_processed": 1,
    "total_timestamps": 87,
    "status": "processing",
    "timestamps": {
        "00:00:00": [
            {
                "main_object": "person",
                "frame_number": 1,
                "results": {
                    "main": {...},
                    "crop": [
                        {
                            "drc": "descripción detallada del crop",
                            "od": {"bboxes": [...], "labels": [...]},
                            "crop_index": 0,
                            "label": "person"
                        }
                    ]
                }
            }
        ]
    }
}
```

### Mensaje Final (cuando se completa todo)
```json
{
    "timestamp": "2024-01-01T12:00:00",
    "partial": false,
    "complete": true,
    "full_minio_path": "my-bucket/video-id/job-id/fine_detections/complete.json"
}
```

## Recomendaciones de Uso

1. **Para desarrollo/debug**: Use `CROPS_ONLY=true MAX_TIMESTAMPS=1`
2. **Para procesamiento rápido de crops**: Use `CROPS_ONLY=true PARALLEL_PROCESSING=true`
3. **Para producción con recursos limitados**: Use `MAX_CROPS_PER_IMAGE=10` y `SCALE_FACTOR=4.0`
4. **Para máxima velocidad**: Use todos los cores con `CROPS_ONLY=true MAX_WORKERS=0`
5. **Para máxima calidad**: Use configuración por defecto (FULL mode, scale 8x)

## Procesamiento Paralelo

El sistema ahora puede procesar múltiples crops en paralelo:
- Por defecto usa el 80% de los CPUs disponibles
- Cada crop se procesa independientemente
- Los resultados se envían a Kafka tan pronto como se completan
- Reduce significativamente el tiempo de procesamiento

## Monitoreo

Los logs ahora incluyen:
- `[MinIO]` - Tiempos de descarga
- `[LLM]` - Tiempos de inferencia y dispositivo usado (CPU/GPU)
- `[KAFKA]` - Mensajes enviados
- `[UPDATE]` - Cuando se actualiza complete.json
- `[ITEM X/Y]` - Progreso dentro de cada timestamp
- `[TIMESTAMP X/Y]` - Progreso general
- `[PARALLEL]` - Información sobre procesamiento paralelo

## Consumir Resultados desde el Frontend

1. **Polling del complete.json**: 
   - El frontend puede hacer polling al mismo archivo `complete.json`
   - Verificar el campo `status` para saber si terminó
   - Los resultados se van acumulando en el campo `timestamps`

2. **Escuchar mensajes Kafka**:
   - Cada mensaje incluye la ruta al `complete.json` actualizado
   - No necesitas procesar mensajes individuales, solo recargar el complete.json