from libs.queues import KafkaHandler
import json
from minio import Minio
import os
from typing import List


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
    kafka_address = '192.168.1.12:9093'
    kafka_handler = KafkaHandler(bootstrap_servers=[kafka_address])
    
    print("Consuming... ")
    messages = ["video:5049e5f6-ec91-4afb-b2f5-a63a991a7993_label:complete"]

    for message in messages:
        print("Sending...")
        # Create a producer and send a message
        kafka_handler.produce_message('FINE_TASK', message)
        