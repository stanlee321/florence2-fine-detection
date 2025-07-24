from libs.core import Application
import os
from dotenv import load_dotenv
import sys

load_dotenv()  

SERVER_IP = os.getenv('IP_ADDRESS')
API_BASE_URL = f"http://{SERVER_IP}:8003"
minio_key = os.getenv('MINIO_ACCESS_KEY')
minio_secret = os.getenv('MINIO_SECRET_KEY')
minio_url = f"{SERVER_IP}:9000"
llm_model_id = "microsoft/Florence-2-base-ft"
brokers = [f'{SERVER_IP}:9092']
redis_host = f'{SERVER_IP}'
TOPIC_OUTPUT = "video-ls"


BACKEND_EMAIL =  os.getenv("BACKEND_EMAIL", "admin@example.com")
BACKEND_PASSWORD = os.getenv("BACKEND_PASSWORD", "Adminpassword1@")
BACKEND_BASE_URL = f"http://{SERVER_IP}:3001"


if __name__ == '__main__':
    # Check for debug mode
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
    MAX_TIMESTAMPS = int(os.getenv('MAX_TIMESTAMPS', '0'))  # 0 = process all
    PROCESS_MODE = os.getenv('PROCESS_MODE', 'FULL').upper()  # FULL, FAST, SAMPLE
    MAX_CROPS_PER_IMAGE = int(os.getenv('MAX_CROPS_PER_IMAGE', '0'))  # 0 = process all
    SCALE_FACTOR = float(os.getenv('SCALE_FACTOR', '8.0'))  # Default 8x
    
    if DEBUG_MODE:
        print("\n[DEBUG MODE ENABLED]")
        if MAX_TIMESTAMPS > 0:
            print(f"[DEBUG] Will process only first {MAX_TIMESTAMPS} timestamps")
        print(f"[DEBUG] Process mode: {PROCESS_MODE}")
        if MAX_CROPS_PER_IMAGE > 0:
            print(f"[DEBUG] Max crops per image: {MAX_CROPS_PER_IMAGE}")
        print(f"[DEBUG] Scale factor: {SCALE_FACTOR}x")
        print("\n")

    # Kafka topics
    topic_input = 'video-input-details'
    bucket_name = "my-bucket"


    app = Application(bucket_name = bucket_name, 
                      minio_host = minio_url,
                      minio_access_key = minio_key,
                      minio_secret_key = minio_secret,
                      llm_model_id = llm_model_id,
                      bootstrap_servers = brokers,
                      redis_host = redis_host,
                      topic_output = TOPIC_OUTPUT,
                      backend_base_url = BACKEND_BASE_URL,
                      backend_email = BACKEND_EMAIL,
                      backend_password = BACKEND_PASSWORD)
    
    # prompt_color_lambda = lambda x: f"<COLOR_DETECTION> The {x} is"

    app.run(offset = 'latest', topic_input = topic_input)
    print("Done")
