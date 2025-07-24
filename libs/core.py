
import re
import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from typing import List, Union
from minio import Minio
import requests
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# MinIO
from libs.queues import KafkaHandler
from libs.redis_service import RedisClient
from libs.api import UpdateStatus
from libs.llm import LLMHandler


class Application:
    def __init__(self, 
                 minio_host: str,
                 minio_access_key: str ,
                 minio_secret_key: str ,
                 bootstrap_servers: List[str],
                 redis_host: str,
                 output_dir: str = "./tmp", 
                 bucket_name: str = "my-bucket",
                 llm_model_id: str = "microsoft/Florence-2-base-ft",
                 topic_output: str = "video-summary",
                 backend_base_url: str = "https://backend.ai.drc-ai.com",
                 backend_email: str = "admin@drc-ai.com",
                 backend_password: str = "admin",
                 ):

        self.client_minio = Minio(minio_host,
                                    access_key=minio_access_key,
                                    secret_key=minio_secret_key,
                                    secure=False)

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.workdir = None

        # Redis Client
        self.redis_client = RedisClient(host=redis_host, port=6379, password="Secret.")
        # kafka handler
        self.kafka_handler = KafkaHandler(
            bootstrap_servers=bootstrap_servers)
        
        # LLM Handler
        self.llm_handler = LLMHandler(model_id=llm_model_id)

        # Prompts en español
        self.od_prompt = "<OD>"
        self.drc_prompt = "Describe detalladamente en español lo que ves en esta imagen"
        
        # Diccionario de traducción de etiquetas
        self.label_translations = {
            "person": "persona",
            "car": "auto",
            "truck": "camión",
            "bus": "autobús",
            "motorcycle": "motocicleta",
            "bicycle": "bicicleta",
            "backpack": "mochila",
            "handbag": "bolso",
            "suitcase": "maleta",
            "bottle": "botella",
            "cup": "taza",
            "fork": "tenedor",
            "knife": "cuchillo",
            "spoon": "cuchara",
            "bowl": "tazón",
            "banana": "plátano",
            "apple": "manzana",
            "sandwich": "sándwich",
            "orange": "naranja",
            "broccoli": "brócoli",
            "carrot": "zanahoria",
            "hot dog": "hot dog",
            "pizza": "pizza",
            "donut": "dona",
            "cake": "pastel",
            "chair": "silla",
            "couch": "sofá",
            "potted plant": "planta en maceta",
            "bed": "cama",
            "dining table": "mesa de comedor",
            "toilet": "inodoro",
            "tv": "televisor",
            "laptop": "laptop",
            "mouse": "ratón",
            "remote": "control remoto",
            "keyboard": "teclado",
            "cell phone": "teléfono celular",
            "microwave": "microondas",
            "oven": "horno",
            "toaster": "tostadora",
            "sink": "lavabo",
            "refrigerator": "refrigerador",
            "book": "libro",
            "clock": "reloj",
            "vase": "florero",
            "scissors": "tijeras",
            "teddy bear": "oso de peluche",
            "hair drier": "secador de pelo",
            "toothbrush": "cepillo de dientes",
            "sneakers": "zapatillas",
            "walking_shoe": "zapato para caminar",
            "shoe": "zapato",
            "hat": "sombrero",
            "cap": "gorra",
            "sunglasses": "lentes de sol",
            "bag": "bolsa",
            "tie": "corbata",
            "suitcase": "maleta",
            "frisbee": "frisbee",
            "skis": "esquís",
            "snowboard": "tabla de snowboard",
            "sports ball": "pelota deportiva",
            "kite": "cometa",
            "baseball bat": "bate de béisbol",
            "baseball glove": "guante de béisbol",
            "skateboard": "patineta",
            "surfboard": "tabla de surf",
            "tennis racket": "raqueta de tenis",
            "umbrella": "paraguas",
            "wheel": "rueda",
            "suv": "camioneta",
            "minivan": "minivan",
            "sedan": "sedán",
            "van": "furgoneta",
            "pickup truck": "camioneta pickup",
            "traffic light": "semáforo",
            "fire hydrant": "hidrante",
            "stop sign": "señal de alto",
            "parking meter": "parquímetro",
            "bench": "banca",
            "bird": "pájaro",
            "cat": "gato",
            "dog": "perro",
            "horse": "caballo",
            "sheep": "oveja",
            "cow": "vaca",
            "elephant": "elefante",
            "bear": "oso",
            "zebra": "cebra",
            "giraffe": "jirafa"
        }
        self.bucket_name = bucket_name
        
        self.topic_output = topic_output
        
        self.updater = UpdateStatus(backend_base_url, backend_email, backend_password)

        
    def set_names(self, video_id: str):
        self.workdir = os.path.join(self.output_dir, video_id)
        os.makedirs(self.workdir, exist_ok=True)

    def set_end_json(self, timestamp: str,):
        timestamp_data = os.path.join(self.workdir, f"{timestamp}.json")
        return timestamp_data

    def get_local_path(self, remote_path: str):
        image_path = os.path.join(self.workdir, remote_path.split("/")[-1])
        return image_path
    
    def translate_label(self, label: str) -> str:
        """Traduce etiquetas de inglés a español"""
        # Convertir a minúsculas para buscar
        label_lower = label.lower()
        # Si existe traducción, usarla. Si no, devolver original
        return self.label_translations.get(label_lower, label)

    @staticmethod
    def parse_string(input_string):
        print("Parsing string...")
        print(input_string)
        print("--------------------------------")
        pattern = r"video:(?P<video>[a-zA-Z0-9\-]+)_label:(?P<label>\w+)"
        match = re.match(pattern, input_string)

        if match:
            video = match.group("video")
            label = match.group("label")
            return video, label
        else:
            raise ValueError(
                "Input string does not match the expected format.")

    @staticmethod
    def extract_relevant_data(parsed_data):
        extracted_data = {}
        for timestamp, details in parsed_data['ground_detections'].items():
            extracted_data[timestamp] = []
            for item in details['data']:
                extracted_data[timestamp].append({
                    "s3_path": item["s3_path"],
                    "fps": item["fps"],
                    "total_frames": item["total_frames"],
                    "frame_number": item["frame_number"],
                    "annotated_video": item["annotated_video"],
                    "name": item["name"],
                    "box": {
                        "x1": item["box.x1"],
                        "y1": item["box.y1"],
                        "x2": item["box.x2"],
                        "y2": item["box.y2"]
                    }
                })
        return extracted_data

    def crop_image(self, image: Union[np.ndarray, str], box: dict, padding: int = 0, from_llm: bool = False,):
        if isinstance(image, str):
            image = cv2.imread(image)
        if from_llm:
            x1, y1, x2, y2 = map(int, [box[0], box[1], box[2], box[3]])
        else:
            x1, y1, x2, y2 = map(int, [box['x1'], box['y1'], box['x2'], box['y2']])
            
        # Add padding to the bounding box to extend the end image so we can draw the grid and labels in a more
        # clear way
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        
        cropped_image = image[y1:y2, x1:x2]
        return cropped_image

    @staticmethod
    def draw_grid_bboxes(image: Union[np.ndarray, str], bboxes, labels, grid_color=(255, 255, 255), line_thickness=1, grid_spacing=50, font_scale=1.0, padding = 0):

        if isinstance(image, str):
            image = cv2.imread(image)

        output_image = image.copy()
        for bbox, label in zip(bboxes, labels):
            start_point = (int(bbox[0]), int(bbox[1]))
            end_point = (int(bbox[2]), int(bbox[3]))
            
            # Add padding to the bounding box to extend the end image so we can draw the grid and labels in a more
            # clear way
            start_point = (max(0, start_point[0] - padding), max(0, start_point[1] - padding))
            end_point = (min(image.shape[1], end_point[0] + padding), min(image.shape[0], end_point[1] + padding))
            
            
            # Calculate number of grid lines within the bounding box
            num_lines_horizontal = (
                end_point[1] - start_point[1]) // grid_spacing
            num_lines_vertical = (
                end_point[0] - start_point[0]) // grid_spacing

            # Create a mask for blending
            mask = np.zeros_like(image)

            # Draw the main bounding box and grid lines on the mask
            cv2.rectangle(mask, start_point, end_point,
                          grid_color, line_thickness)
            for i in range(1, num_lines_horizontal):
                y = start_point[1] + i * grid_spacing
                cv2.line(
                    mask, (start_point[0], y), (end_point[0], y), grid_color, line_thickness)
            for j in range(1, num_lines_vertical):
                x = start_point[0] + j * grid_spacing
                cv2.line(
                    mask, (x, start_point[1]), (x, end_point[1]), grid_color, line_thickness)
            bold_of_grid = 0.03
            
            # Blending the grid
            sub_img = output_image[start_point[1]:end_point[1], start_point[0]:end_point[0]]
            mask_sub_img = mask[start_point[1]:end_point[1], start_point[0]:end_point[0]]
            blended = cv2.addWeighted(
                sub_img, 0.75, mask_sub_img, bold_of_grid, 0)
            output_image[start_point[1]:end_point[1],
                         start_point[0]:end_point[0]] = blended

            # Add label with background box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_thickness = 2
            padding_text = 25
            
            
            if font_scale >= 4.0:
                font_thickness = 20
                
            text_size = cv2.getTextSize(
                label, font, font_scale, font_thickness)[0]
            
            label_point = (start_point[0], start_point[1] + padding_text)

            if font_scale >= 4.0:
                label_point = (start_point[0], start_point[1] + padding_text*2)

            
            background_tl = (start_point[0], start_point[1] )
            background_br = (start_point[0] + text_size[0], start_point[1]  + padding_text*2)
            
            if font_scale >= 4.0:
                background_tl = (start_point[0], start_point[1] - padding_text*2)
                background_br = (start_point[0] + text_size[0], start_point[1]  + padding_text*2)
            cv2.rectangle(output_image, background_tl,
                          background_br, (0, 0, 0), -1)
            cv2.putText(output_image, label, label_point, font,
                        font_scale, (255, 255, 255), font_thickness)

        return output_image

    @staticmethod
    def enhance_image(image: np.ndarray, scale_factor: float = 1.0):
        # Aumentar el tamaño de la imagen recortada para una mejor visualización
        resized_cropped_image = cv2.resize(image, (int(image.shape[1] * scale_factor),
                                                   int(image.shape[0] * scale_factor)))

        # Other operations
        return resized_cropped_image

    def process_details(self, 
                        bucket_name: str,
                        details: List[dict],
                        timestamp: str,
                        video_id: str,
                        job_id: str,
                        callback_fn=None):

        image_results = []
        detected_image_track = []
        
        print(f"[process_details] Starting to process {len(details)} items for timestamp {timestamp}")

        for item_idx, item in enumerate(details):
            print(f"\n[ITEM {item_idx+1}/{len(details)}] Processing item...")
            # A lot of images will be repeated, so we need to avoid processing them again
            remote_patch = item["s3_path"]

            if remote_patch in detected_image_track:
                continue

            # fps = item["fps"]
            # total_frames = item["total_frames"]
            # track_id = item['track_id']

            # TODO, DO SOMETHING WITH THE MAIN OBJECT BOX
            box = item["box"]
            frame_number = item["frame_number"]
            name = item['name']

            # Download the a
            image_path = self.get_local_path(remote_patch)
            print(f"[MinIO] Downloading image: {remote_patch}")
            download_start = time.time()
            self.client_minio.fget_object(
                bucket_name, remote_patch, image_path)
            download_time = time.time() - download_start
            print(f"[MinIO] Downloaded in {download_time:.2f}s")

            # Check if we should skip main image processing
            CROPS_ONLY = os.getenv('CROPS_ONLY', 'false').lower() == 'true'
            
            if CROPS_ONLY:
                print(f"[CROPS_ONLY] Skipping main image DRC")
                drc_main = {'<MORE_DETAILED_CAPTION>': 'Skipped - CROPS_ONLY mode'}
            else:
                # Describe whole image first.
                print(f"[LLM] Starting DRC (detailed caption) for main image...")
                drc_main = self.llm_handler.describe_image(
                    image_path, task_prompt=self.drc_prompt)

            # OD in the whole image
            print(f"[LLM] Starting OD (object detection) for main image...")
            od_main = self.llm_handler.describe_image(
                image_path, task_prompt=self.od_prompt)

            process_results = {
                "main_object": name,
                "frame_number": frame_number,
                "main_image": remote_patch,
                "results": {
                    "main": {
                        "drc_main": drc_main.get(self.drc_prompt, drc_main.get('<MORE_DETAILED_CAPTION>', '')),
                        "od_main": od_main['<OD>'],
                        "annotated_image_path": None,
                    },
                    "crop": []
                },
                "video_id": video_id,
                "timestamp": timestamp,

            }
            
            od_main_od = od_main['<OD>']
            if len(od_main_od['bboxes']) > 0:
                # Draw the bounding boxes
                image_annotated = self.draw_grid_bboxes(
                    image_path, od_main_od['bboxes'], od_main_od['labels'])

                # Create a path for the annotated image
                annotated_image_path = os.path.join(
                    self.workdir, f"{timestamp}_frame_{frame_number}_annotated.jpg")

                # Save the annotated image
                cv2.imwrite(annotated_image_path, image_annotated)
                annotated_image_path_remote = f"{video_id}/{job_id}/images/annotated_images/{timestamp}_frame_{frame_number}_annotated.jpg"
                # Upload the annotated image to MinIO
                self.client_minio.fput_object(
                    bucket_name, annotated_image_path_remote , annotated_image_path)
                process_results["results"]["main"]["annotated_image_path"] = annotated_image_path_remote
                
                # for each bbox in the OD, crop the image and do OD
                od_results_crop = []
                
                # Get configuration from environment
                PROCESS_MODE = os.getenv('PROCESS_MODE', 'FULL').upper()
                MAX_CROPS_PER_IMAGE = int(os.getenv('MAX_CROPS_PER_IMAGE', '0'))
                scale_factor = float(os.getenv('SCALE_FACTOR', '8.0'))
                padding = 10
                
                # Check if we should process crops
                CROPS_ONLY = os.getenv('CROPS_ONLY', 'false').lower() == 'true'
                
                if PROCESS_MODE == 'FAST' and not CROPS_ONLY:
                    print(f"[FAST MODE] Skipping {len(od_main_od['bboxes'])} crops")
                    process_results["results"]["crop"] = []
                else:
                    # Process crops (main functionality)
                    print(f"[CROPS] Processing {len(od_main_od['bboxes'])} detected objects...")
                    bboxes_to_process = list(zip(od_main_od['bboxes'], od_main_od['labels']))
                    
                    # Limit crops if configured
                    if MAX_CROPS_PER_IMAGE > 0 and len(bboxes_to_process) > MAX_CROPS_PER_IMAGE:
                        print(f"[LIMIT] Processing only first {MAX_CROPS_PER_IMAGE} of {len(bboxes_to_process)} crops")
                        bboxes_to_process = bboxes_to_process[:MAX_CROPS_PER_IMAGE]
                    
                    # Check if parallel processing is enabled
                    PARALLEL_PROCESSING = os.getenv('PARALLEL_PROCESSING', 'true').lower() == 'true'
                    MAX_WORKERS = int(os.getenv('MAX_WORKERS', '0'))
                    
                    if MAX_WORKERS == 0:
                        # Use 80% of available CPUs
                        MAX_WORKERS = max(1, int(multiprocessing.cpu_count() * 0.8))
                    
                    print(f"[PARALLEL] Using {MAX_WORKERS} workers for crop processing")
                    
                    if PARALLEL_PROCESSING and len(bboxes_to_process) > 1:
                        # Prepare arguments for parallel processing
                        crop_args = []
                        for index, (bbox_main, label_main) in enumerate(bboxes_to_process):
                            args = (
                                image_path, bbox_main, label_main, index, timestamp,
                                frame_number, video_id, job_id, bucket_name,
                                scale_factor, padding, len(bboxes_to_process)
                            )
                            crop_args.append(args)
                        
                        # Process crops in parallel
                        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                            futures = []
                            for args in crop_args:
                                future = executor.submit(self.process_single_crop, args)
                                futures.append(future)
                            
                            # Collect results as they complete
                            for future in as_completed(futures):
                                result = future.result()
                                if result:
                                    od_results_crop.append(result)
                                    
                                    # Don't send individual crop callbacks in parallel mode
                                    # They will be included in the main image result
                    else:
                        # Sequential processing (original code)
                        for index, (bbox_main, label_main) in enumerate(bboxes_to_process):
                            args = (
                                image_path, bbox_main, label_main, index, timestamp,
                                frame_number, video_id, job_id, bucket_name,
                                scale_factor, padding, len(bboxes_to_process)
                            )
                            result = self.process_single_crop(args)
                            if result:
                                od_results_crop.append(result)
                        
                    process_results["results"]["crop"] = od_results_crop
                
            detected_image_track.append(remote_patch)
            image_results.append(process_results)
            
            # Call the callback function after each image is processed
            if callback_fn:
                callback_fn(item_idx, len(details), timestamp, process_results)

        return image_results

    def process_single_crop(self, args):
        """Process a single crop - designed to be called in parallel"""
        (
            image_path, bbox_main, label_main, index, timestamp, 
            frame_number, video_id, job_id, bucket_name, 
            scale_factor, padding, total_crops
        ) = args
        
        try:
            label_main = label_main.replace(" ", "_").replace("/", "_")
            results_ci = {}
            
            # Load image
            image = cv2.imread(image_path)
            
            # Crop the image
            cropped_image = self.crop_image(
                image, bbox_main, from_llm=True, padding=padding
            )
            
            # Enhance the image
            cropped_image = self.enhance_image(
                cropped_image, scale_factor=scale_factor
            )
            
            # Do OD in the cropped image
            label_es = self.translate_label(label_main)
            print(f"[LLM] Processing crop {index+1}/{total_crops} - {label_es}")
            od_ci = self.llm_handler.describe_image(
                cropped_image, task_prompt=self.od_prompt
            )
            
            od_ci_od = od_ci['<OD>']
            
            # Calculate font scale based on image size
            def calculate_font_scale(image_size, font_scale=2):
                new_scale = font_scale * (image_size[0] / 1000)
                if new_scale > 10.0:
                    return 4.0
                return new_scale
            
            font_scale = calculate_font_scale(cropped_image.shape)
            
            # Draw the bounding boxes
            image_annotated_bbox_ci = self.draw_grid_bboxes(
                cropped_image,
                od_ci_od['bboxes'], 
                od_ci_od['labels'], 
                font_scale=font_scale, 
                padding=padding + 30
            )
            
            # Do drc in the cropped image
            drc_ci = self.llm_handler.describe_image(
                cropped_image, task_prompt=self.drc_prompt
            )
            
            local_annotated_path_ci_base = f"{timestamp}_frame_{frame_number}_bbox_{index}_label_{label_main}_annotated_ci.jpg"
            annotated_image_path_ci = os.path.join(
                self.workdir, local_annotated_path_ci_base
            )
            
            # Save the annotated image
            cv2.imwrite(annotated_image_path_ci, image_annotated_bbox_ci)
            
            remote_annotated_path_ci = f"{video_id}/{job_id}/images/annotated_images_ci/{local_annotated_path_ci_base}"
            
            # Upload the annotated image to MinIO
            self.client_minio.fput_object(
                bucket_name, remote_annotated_path_ci, annotated_image_path_ci
            )
            
            results_ci["drc"] = drc_ci.get(self.drc_prompt, drc_ci.get('<MORE_DETAILED_CAPTION>', ''))
            results_ci["od"] = od_ci_od
            results_ci["annotated_image"] = local_annotated_path_ci_base
            results_ci["annotated_image_remote"] = remote_annotated_path_ci
            results_ci["crop_index"] = index
            results_ci["label"] = label_main
            
            return results_ci
            
        except Exception as e:
            print(f"[ERROR] Failed to process crop {index}: {str(e)}")
            return None
    
    def draw_bounding_boxes(self, image_path, boxes):
        image = cv2.imread(image_path)
        for box in boxes:
            x1, y1, x2, y2 = map(
                int, [box['x1'], box['y1'], box['x2'], box['y2']])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return image

    def download_file_using_requests(self, url, local_path):
        response = requests.get(url)
        with open(local_path, 'wb') as f:
            f.write(response.content)
    
    def process_message(self, message):
        input_message: dict = message.value
        
        print("Processing message...")
        print(input_message)
        print("--------------------------------")
        
        video_id = input_message['video_id']
        job_id = input_message['job_id']
        model_id = input_message['model_id']
        full_data = input_message['full_data']
        
        self.updater.run(job_id, "Processing")
        
        self.set_names(video_id)

        # Parse the input key
        data_path = f"{self.workdir}/{video_id}.json"
        self.download_file_using_requests(full_data, data_path)
        
        with open(data_path, 'r') as f:
            output_value = json.load(f)
        
        print(output_value.keys())
        # Extract the relevant data
        extracted_data = self.extract_relevant_data(output_value)

        detected_image_track = {
            timestamp: {} for timestamp in extracted_data.keys()
        }
        self.set_names(video_id=video_id)
        
        # Initialize complete results dictionary with flat structure
        complete_results = {
            "data": [],
            "batch": {
                "currentBatch": 1,
                "batchSize": 1000,
                "totalBatches": 0,
                "totalItems": 0,
                "startIndex": 0,
                "endIndex": 0,
                "itemsInBatch": 0,
                "remainingItems": 0,
                "hasMore": False,
                "nextBatch": None,
                "nextStartFrom": None,
                "searchTerm": None,
                "processingStats": {
                    "totalTimestamps": len(extracted_data),
                    "totalFrames": 0,
                    "filteredItems": 0
                }
            },
            "metadata": {
                "video_id": video_id,
                "job_id": job_id,
                "model_id": model_id
            }
        }

        total_timestamps = len(extracted_data.items())
        print(f"\n[INFO] Processing {total_timestamps} timestamps...")
        
        # Check if we should limit processing (debug mode)
        DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
        MAX_TIMESTAMPS = int(os.getenv('MAX_TIMESTAMPS', '0'))
        
        items_to_process = list(extracted_data.items())
        if DEBUG_MODE and MAX_TIMESTAMPS > 0:
            items_to_process = items_to_process[:MAX_TIMESTAMPS]
            print(f"[DEBUG] Limiting to first {MAX_TIMESTAMPS} timestamps")
        
        for timestamp_idx, (timestamp, details) in enumerate(tqdm(items_to_process, desc="Timestamps")):
            print(f"\n[TIMESTAMP {timestamp_idx+1}/{total_timestamps}] Processing timestamp: {timestamp}")
            print(f"[TIMESTAMP {timestamp}] Number of details to process: {len(details)}")
            
            # Create callback to send results incrementally
            def send_incremental_result(item_idx, total_items, timestamp, result):
                # Process result to flat format if it contains crops
                if "results" in result and "crop" in result["results"]:
                    for crop_idx, crop_data in enumerate(result["results"]["crop"]):
                        flat_item = {
                            "timestamp": 0,  # Will need to convert timestamp to number
                            "frameIndex": result.get("frame_number", 0),
                            "cropIndex": crop_idx,
                            "itemId": f"{timestamp}_{result.get('frame_number', 0)}_{crop_idx}",
                            "description": crop_data.get("drc", ""),
                            "confidence": 0,
                            "bbox": crop_data.get("od", {}).get("bboxes", []),
                            "label": self.translate_label(crop_data.get("label", "")),
                            "fullFrame": result
                        }
                        complete_results["data"].append(flat_item)
                else:
                    # If no crops, add main result
                    flat_item = {
                        "timestamp": 0,
                        "frameIndex": result.get("frame_number", 0),
                        "cropIndex": 0,
                        "itemId": f"{timestamp}_{result.get('frame_number', 0)}_0",
                        "description": result.get("results", {}).get("main", {}).get("drc_main", ""),
                        "confidence": 0,
                        "bbox": [],
                        "label": self.translate_label(result.get("main_object", "")),
                        "fullFrame": result
                    }
                    complete_results["data"].append(flat_item)
                
                # Update batch statistics
                complete_results["batch"]["totalItems"] = len(complete_results["data"])
                complete_results["batch"]["itemsInBatch"] = len(complete_results["data"])
                complete_results["batch"]["endIndex"] = len(complete_results["data"]) - 1
                
                # Save and upload updated complete.json
                complete_json_path = os.path.join(self.workdir, "complete.json")
                with open(complete_json_path, 'w') as f:
                    json.dump(complete_results, f, indent=2)
                
                # Upload/overwrite complete.json in MinIO
                complete_remote_path = f"{video_id}/{job_id}/fine_detections/complete.json"
                self.client_minio.fput_object(
                    self.bucket_name, complete_remote_path, complete_json_path)
                
                print(f"[UPDATE] Updated complete.json with item {item_idx+1}/{total_items} for timestamp {timestamp}")
                
                # Also save single result for reference
                single_result_path = os.path.join(self.workdir, f"{timestamp}_item_{item_idx}.json")
                with open(single_result_path, 'w') as f:
                    json.dump({
                        "timestamp": timestamp,
                        "item_index": item_idx,
                        "total_items": total_items,
                        "data": result
                    }, f)
                
                # Send to Kafka with complete.json path
                self.kafka_handler.produce_message(
                    topic=self.topic_output,
                    message=json.dumps({
                        "timestamp": timestamp,
                        "item_index": item_idx,
                        "total_items": total_items,
                        "partial": True,
                        "complete_json_path": f"{self.bucket_name}/{complete_remote_path}",
                        "full_minio_path": f"{self.bucket_name}/{complete_remote_path}"
                    }))
                print(f"[KAFKA] Sent partial result {item_idx+1}/{total_items} for timestamp {timestamp}")
            
            llm_results = self.process_details(
                self.bucket_name,
                details,
                timestamp,
                video_id,
                job_id,
                callback_fn=send_incremental_result)

            timestamp_data_path = self.set_end_json(timestamp)

            detected_image_track[timestamp] = llm_results
            
            # No need to update here since callback already updated complete_results
            # Just update final statistics
            complete_results["batch"]["processingStats"]["totalFrames"] += len(llm_results)

            # Save the descriptions to a JSON file
            with open(timestamp_data_path, 'w') as f:
                json.dump(detected_image_track, f)

            #########################################################
            # Save final complete.json with all timestamps processed so far
            complete_json_path = os.path.join(self.workdir, "complete.json")
            complete_results["metadata"]["total_timestamps_processed"] = timestamp_idx + 1
            complete_results["metadata"]["total_timestamps"] = len(items_to_process)
            complete_results["metadata"]["status"] = "processing" if timestamp_idx + 1 < len(items_to_process) else "complete"
            
            # Update final batch stats
            complete_results["batch"]["totalBatches"] = 1
            complete_results["batch"]["hasMore"] = timestamp_idx + 1 < len(items_to_process)
            
            with open(complete_json_path, 'w') as f:
                json.dump(complete_results, f, indent=2)
            
            # Upload the complete data to MinIO
            json_remote_path = f"{video_id}/{job_id}/fine_detections/complete.json"
            self.client_minio.fput_object(
                self.bucket_name, json_remote_path, complete_json_path)


            #########################################################
            # Single data and local path
            single_data_path = os.path.join(self.workdir, f"{timestamp}.json")
            with open(single_data_path, 'w') as f:
                json.dump({
                    "timestamp": timestamp,
                    "data": llm_results
                    }, f)
            self.client_minio.fput_object(
                self.bucket_name, f"{video_id}/{job_id}/fine_detections/{timestamp}.json", single_data_path)
            
            # Full minio path
            full_minio_path = f"{self.bucket_name}/{video_id}/{job_id}/fine_detections/{timestamp}.json"
            #########################################################
            # Send final complete message to Kafka
            self.kafka_handler.produce_message(
                topic=self.topic_output,
                message=json.dumps({
                    "timestamp": timestamp,
                    "partial": False,
                    "complete": True,
                    "full_minio_path": full_minio_path
                }))
            print(f"[KAFKA] Sent COMPLETE result for timestamp {timestamp}")
            
        # Only mark as finished after all timestamps are processed
        self.updater.run(job_id, "Finished")


    def get_uuid(self):
        return str(uuid.uuid4())
    
    def run(self, offset: str = 'latest', topic_input: str = 'fine-detections'):
        print("Waiting for messages...")
        topic_group = f'fine-group-{self.get_uuid()}'
        # # Create Kafka consumer
        consumer = self.kafka_handler.create_consumer(topic_input, topic_group, auto_offset_reset=offset)

        for message in consumer:
            print(f"Consumed message: {message.value}")
            self.process_message(message)
