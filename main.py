from libs.core import Application
import uuid

SERVER_IP = "192.168.1.19"
API_BASE_URL = f"http://{SERVER_IP}:8003"
minio_key = "c3aFDmKuGhPCSxkpRDGf"
minio_secret = "MLYz9tZI3h4xZAwBl8llyEtX6R07YcMuRdSYPIcx"
minio_url = f"{SERVER_IP}:9000"
llm_model_id = "microsoft/Florence-2-base-ft"
brokers = [f'{SERVER_IP}:9092']
redis_host = f'{SERVER_IP}'


if __name__ == '__main__':


    # Kafka topics
    topic_input = 'fine-detections'
    bucket_name = "my-bucket"


    app = Application(bucket_name = bucket_name, 
                      minio_host = minio_url,
                      minio_access_key = minio_key,
                      minio_secret_key = minio_secret,
                      llm_model_id = llm_model_id,
                      bootstrap_servers = brokers,
                      redis_host = redis_host)
    
    # prompt_color_lambda = lambda x: f"<COLOR_DETECTION> The {x} is"

    app.run(offset = 'latest', topic_input = topic_input)
    print("Done")
