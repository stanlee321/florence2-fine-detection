
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

# MinIO
from libs.queues import KafkaHandler
from libs.redis_service import RedisClient

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

        self.od_prompt = "<OD>"
        self.drc_prompt = "<MORE_DETAILED_CAPTION>"
        self.bucket_name = bucket_name
        
        self.topic_output = topic_output
        
    def set_names(self, video_id: str):
        self.workdir = os.path.join(self.output_dir, video_id)
        os.makedirs(self.workdir, exist_ok=True)

    def set_end_json(self, timestamp: str,):
        timestamp_data = os.path.join(self.workdir, f"{timestamp}.json")
        return timestamp_data

    def get_local_path(self, remote_path: str):
        image_path = os.path.join(self.workdir, remote_path.split("/")[-1])
        return image_path

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
                        video_id: str):

        image_results = []
        detected_image_track = []

        for item in details:
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
            self.client_minio.fget_object(
                bucket_name, remote_patch, image_path)

            # Describe whole image first.
            drc_main = self.llm_handler.describe_image(
                image_path, task_prompt=self.drc_prompt)

            # OD in the whole image
            od_main = self.llm_handler.describe_image(
                image_path, task_prompt=self.od_prompt)

            process_results = {
                "main_object": name,
                "frame_number": frame_number,
                "main_image": remote_patch,
                "results": {
                    "main": {
                        "drc_main": drc_main['<MORE_DETAILED_CAPTION>'],
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
                annotated_image_path_remote = f"{video_id}/images/annotated_images/{timestamp}_frame_{frame_number}_annotated.jpg"
                # Upload the annotated image to MinIO
                self.client_minio.fput_object(
                    bucket_name, annotated_image_path_remote , annotated_image_path)
                process_results["results"]["main"]["annotated_image_path"] = annotated_image_path_remote
                
                # for each bbox in the OD, crop the image and do OD
                od_results_crop = []
                scale_factor = 8
                padding = 10
                for index , (bbox_main, label_main) in enumerate(zip(od_main_od['bboxes'], od_main_od['labels'])):
                    label_main = label_main.replace(" ", "_").replace("/", "_")
                    results_ci = {}
                    
                    # Crop the image and store the results in a dictionary
                    cropped_image = self.crop_image(
                        image_path, 
                        bbox_main, from_llm=True,
                        padding=padding,
                        )

                    # Enhance the image
                    cropped_image = self.enhance_image(
                        cropped_image, scale_factor=scale_factor)
                    
                    # Do OD in the croped image
                    od_ci = self.llm_handler.describe_image(
                        cropped_image, task_prompt=self.od_prompt)
                    

                    od_ci_od = od_ci['<OD>']
                    
                    # Calculate fornt scale based on image size
                    def calculate_font_scale(image_size, font_scale=2):
                        new_scale =  font_scale * (image_size[0] / 1000)
                        if new_scale > 10.0:
                            return 4.0
                        return new_scale
                    font_scale = calculate_font_scale(cropped_image.shape)
                    # print("Font scale: ", font_scale)
                    # Draw the bounding boxes
                    image_annotated_bbox_ci = self.draw_grid_bboxes(
                        cropped_image,
                        od_ci_od['bboxes'], 
                        od_ci_od['labels'], 
                        font_scale=font_scale, 
                        padding=padding + 30)  

                    # Do drc in the croped image
                    drc_ci = self.llm_handler.describe_image(
                        cropped_image, task_prompt=self.drc_prompt) 

                    local_annotated_path_ci_base = f"{timestamp}_frame_{frame_number}_bbox_{index}_label_{label_main}_annotated_ci.jpg"
                    # Create a path for the annotated image
                    annotated_image_path_ci = os.path.join(
                        self.workdir, local_annotated_path_ci_base)


                    # Save the annotated image
                    cv2.imwrite(annotated_image_path_ci,
                                image_annotated_bbox_ci)
                    
                    remote_annotated_path_ci = f"{video_id}/images/annotated_images_ci/{local_annotated_path_ci_base}"
                    # Upload the annotated image to MinIO
                    self.client_minio.fput_object(
                        bucket_name,remote_annotated_path_ci , annotated_image_path_ci)

                    results_ci["drc"] = drc_ci['<MORE_DETAILED_CAPTION>']
                    results_ci["od"] = od_ci_od
                    results_ci["annotated_image"] = local_annotated_path_ci_base
                    results_ci["annotated_image_remote"] = remote_annotated_path_ci
                    
                    od_results_crop.append(results_ci)
                    
                process_results["results"]["crop"] = od_results_crop
                
            detected_image_track.append(remote_patch)
            image_results.append(process_results)

        return image_results

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
        video_id = input_message['general_id']
        job_id = input_message['job_id']
        model_id = input_message['model_id']
        full_data = input_message['full_data']
        
        self.set_names(video_id)

        # Parse the input key
        data_path = f"{self.workdir}/{video_id}.json"
        self.download_file_using_requests(full_data, data_path)
        
        with open(data_path, 'r') as f:
            output_value = json.load(f)
        
        try:
            output_value:dict = json.loads(output_value)
        except Exception as e:
            print("Error parsing the output value, ", e)
            return

        print(output_value.keys())
        # Extract the relevant data
        extracted_data = self.extract_relevant_data(output_value)

        detected_image_track = {
            timestamp: {} for timestamp in extracted_data.keys()
        }
        self.set_names(video_id=video_id)

        for timestamp, details in tqdm(extracted_data.items()):

            llm_results = self.process_details(
                self.bucket_name,
                details,
                timestamp,
                video_id)

            timestamp_data_path = self.set_end_json(timestamp)

            detected_image_track[timestamp] = llm_results

            # Save the descriptions to a JSON file
            with open(timestamp_data_path, 'w') as f:
                json.dump(detected_image_track, f)

            #########################################################
            # Upload the complete data to MinIO
            self.client_minio.fput_object(
                self.bucket_name, f"{video_id}/fine_detections/complete.json", timestamp_data_path)


            #########################################################
            # Single data and local path
            single_data_path = os.path.join(self.workdir, f"{timestamp}.json")
            with open(single_data_path, 'w') as f:
                json.dump({
                    "timestamp": timestamp,
                    "data": llm_results
                    }, f)
            self.client_minio.fput_object(
                self.bucket_name, f"{video_id}/fine_detections/{timestamp}.json", single_data_path)
            
            #########################################################
            # Send to Kafka the processed data
            self.kafka_handler.produce_message(
                topic_input=self.topic_output,
                message=json.dumps({
                    "timestamp": timestamp,
                    "data": llm_results
                }))

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
