from libs.core import Application
import os
from dotenv import load_dotenv

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


if __name__ == '__main__':


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
                    topic_output = TOPIC_OUTPUT)
    
    # prompt_color_lambda = lambda x: f"<COLOR_DETECTION> The {x} is"

    app.run(offset = 'latest', topic_input = topic_input)
    print("Done")
