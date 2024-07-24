import re
import os
import json

from tqdm import tqdm
from typing import List, Callable
from minio import Minio

# MinIO
from libs.queues import KafkaHandler
from libs.redis_service import RedisClient

from libs.llm import LLMHandler

    
def parse_string(input_string):
    pattern = r"video:(?P<video>[a-zA-Z0-9\-]+)_label:(?P<label>\w+)"
    match = re.match(pattern, input_string)
    
    if match:
        video = match.group("video")
        label = match.group("label")
        return video, label
    else:
        raise ValueError("Input string does not match the expected format.")
    

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


def process_details(llm_handler: LLMHandler, details : List[dict], task_prompt: Callable, main_prompts: List[str]):
    
    image_descriptions = []
    detected_image_track = []
                
    for item in tqdm(details):
        # A lot of images will be repeated, so we need to avoid processing them again
        s3_path = item["s3_path"]
        
        if s3_path in detected_image_track:
            continue
        
        # Placeholder for the LLM results
        llm_results = {
            x: None for x in main_prompts
        }
        
    
        # fps = item["fps"]
        # total_frames = item["total_frames"]
        # track_id = item['track_id']

        box = item["box"]
        frame_number = item["frame_number"]
        name = item['name']
        
        # Download the annotated video from MinIO
        image_path = os.path.join(output_folder, s3_path.split("/")[-1])
        
        client_minio.fget_object(bucket_name, s3_path, image_path)
        
        # Process the frames
        description_results_crop =  \
                    llm_handler.describe_image_on_crop( name,
                                                        box, 
                                                        image_path, 
                                                        task_prompt(name), 
                                                        aux_prompt= "<MORE_DETAILED_CAPTION>")
        
        for prompt in main_prompts:
            llm_results[prompt] = llm_handler.describe_image(image_path, task_prompt=prompt)
        
        data_result = {
            "frame_number": frame_number,
            "s3_path": s3_path,
            "description_general": llm_results,
            "description_results_crop": description_results_crop
        }

        image_descriptions.append(data_result)
        detected_image_track.append(s3_path)

    return image_descriptions


if __name__ == '__main__':
    
    output_folder = './tmp/'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    
    client_minio = Minio("0.0.0.0:9000",  # Replace with your MinIO storage address
        access_key = "minioadmin",   # Replace with your access key
        secret_key = "minioadmin",    # Replace with your secret key
        secure = False
    )
    
    # Redis Client
    redis_client = RedisClient(host='0.0.0.0', port=6379)
    
    # kafka handler
    kafka_handler = KafkaHandler(bootstrap_servers=['192.168.1.12:9093'])
    
    # LLM Handler
    llm_handler = LLMHandler()
    
    
    topic_input = 'FINE_TASK'
    topic_group = 'fine-group'
    bucket_name = "my-bucket"
    

    # # Create Kafka consumer
    consumer = kafka_handler.create_consumer(topic_input, topic_group)


    main_prompts = ["<OD>", 
                    "<MORE_DETAILED_CAPTION>", 
                    # '<DENSE_REGION_CAPTION>', 
                    # '<CAPTION_TO_PHRASE_GROUNDING>'
                ]
    prompt_color_lambda = lambda x: f"<COLOR_DETECTION> The {x} is"

    counter = 0
    
    for message in consumer:
        
        if counter == 2:
            break
        input_key:str = message.value
        
        # Get the output value from Redis
        output_value: dict = redis_client.get_value(input_key)
        
        # Parse the input key
        video, label = parse_string(input_key)
            
        # Extract the relevant data
        extracted_data = extract_relevant_data(output_value)
        
        detected_image_track = {
            timestamp: {} for timestamp in extracted_data.keys()
        }
        
        for timestamp, details in extracted_data.items():

            llm_results = process_details(llm_handler,
                details, 
                task_prompt = prompt_color_lambda, 
                main_prompts = main_prompts)
                
            detected_image_track[timestamp] = llm_results
            
            # Save the descriptions to a JSON file
            print(f"Saving descriptions for {video}_{label}...")
            with open(f"{output_folder}{video}_{label}_{timestamp}_fine.json", 'w') as f:
                json.dump(detected_image_track, f)
            
        counter += 1
        
    print("Done")