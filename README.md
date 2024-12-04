# Project Documentation

## Overview

This project is designed to process video data, leveraging various services such as Kafka for messaging, Redis for caching, and MinIO for object storage. It utilizes a custom Long-Long Model (LLM) handler for image processing and description, integrating with a Kafka queue to manage tasks efficiently.

## Prerequisites

- Python 3.x
- Kafka
- Redis
- MinIO
- Libraries: `redis`, `json`, `os`, `minio`, `tqdm`

## Setup

1. **Kafka Setup**: Ensure Kafka is running and accessible. Update the `kafka_address` in `injest.py` and `main.py` to point to your Kafka server.

2. **Redis Setup**: Ensure Redis is running and accessible. Update the Redis host and port in `libs/redis_service.py` and `main.py` as needed.

3. **MinIO Setup**: Ensure MinIO is running and accessible. Update the MinIO connection details in `main.py` with your MinIO server's address, access key, and secret key.

4. **Install Dependencies**: Install the required Python libraries by running:

   ```sh
   pip install redis minio tqdm
   ```

## Running the Application

1. Start the Consumer: Run main.py to start consuming messages from Kafka. This script processes video data, interacts with Redis, and stores results in MinIO.

2. Ingest Data: Use injest.py to load demo data and send messages to Kafka for processing. This script simulates the ingestion process of video data.

## Architecture

1. KafkaHandler: Manages Kafka topics, producing and consuming messages. See libs/queues.py.

2. RedisClient: Facilitates interaction with Redis for caching purposes. See libs/redis_service.py.

3. LLMHandler: Handles the processing of images using a Long-Long Model. See libs/llm.py.

4. Main Processing: The main script (main.py) orchestrates the consumption of Kafka messages, processing of video data, and interaction with Redis and MinIO.

## Data Processing Flow\*\*

1. Message Consumption:\*\* main.py consumes messages from a Kafka topic, indicating video data to be processed.

2. Data Retrieval and Processing: For each message, the script retrieves related data from Redis, processes it using the LLMHandler, and stores the processed data in MinIO.

3. Result Storage: Processed data, including image descriptions, are stored in MinIO for further use.

## Notebooks

- Florence 2 Notebook: Demonstrates additional processing and visualization in a Jupyter notebook environment. See notebooks/florence_2.ipynb.
  Logging
  Logs are generated and stored in kafka_handler.log for monitoring and debugging purposes.

## Temporary Data

- Processed data and temporary files are stored in the tmp/ directory.

## Contribution

Feel free to contribute to this project by submitting pull requests or opening issues for bugs and feature requests.

FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install flash-attn --no-build-isolation
optimum-cli export onnx --model microsoft/Florence-2-base-ft florence_base_onnx/  --task text-generation --trust-remote-code