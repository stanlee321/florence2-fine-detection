from libs.queues import KafkaHandler
import json
from minio import Minio
import os
from typing import List
from libs.redis_service import RedisClient

with open('demo_data.json', 'r') as f:
    data = json.load(f)
    
def load_demo_data(host_folder: str) -> List[dict]:
    # List files in output/{asset_id} dir
    data_dict = []
    files = os.listdir(host_folder)
    for file in files:
        if file.endswith('.json') and "output_json_timestamp" not in file:
            with open(host_folder + "/" + file, 'r') as f:
                data_dict.append(json.load(f))
    return data_dict


if __name__ == '__main__':

    print("Startingg...")
    kafka_address = 'localhost:9093'
    kafka_handler = KafkaHandler(bootstrap_servers=[kafka_address])
    redis_client = RedisClient(host='0.0.0.0', port=6379)

    key = "video:8d0a09c1-4d74-469e-947f-26d4dcf2bc85_label:complete"
    
    redis_client.set_value(key, json.dumps(data))
    
    print("Consuming... ")
    messages = [
        {
            "lite_data": "video:8d0a09c1-4d74-469e-947f-26d4dcf2bc85_label:lite",
            "full_data": "video:8d0a09c1-4d74-469e-947f-26d4dcf2bc85_label:complete"
        }
    ]

    for message in messages:
        print("Sending...")
        # Create a producer and send a message
        kafka_handler.produce_message('fine-detections', message)
