import os
import json
import argparse
import yaml
import time
import torch
import cv2
import pika
import clickhouse_connect
from dotenv import load_dotenv

from imread_from_url import imread_from_url
from sapiens_inference.segmentation import SapiensSegmentation, SapiensSegmentationType, segmentation_to_polygons, draw_segmentation_map, classes

# Load environment variables
load_dotenv()

def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def get_rabbitmq_connection():
    url = os.environ.get('RABBITMQ_URL', 'amqp://guest:guest@localhost:5672/')
    parameters = pika.URLParameters(url)
    return pika.BlockingConnection(parameters)

def get_clickhouse_client():
    host = os.environ.get('CLICKHOUSE_HOST', 'localhost')
    port = int(os.environ.get('CLICKHOUSE_PORT', '8123'))
    username = os.environ.get('CLICKHOUSE_USER', 'default')
    password = os.environ.get('CLICKHOUSE_PASSWORD', '')
    database = os.environ.get('CLICKHOUSE_DB', 'default')
    return clickhouse_connect.get_client(host=host, port=port, username=username, password=password, database=database)

def process_batch(channel, messages, estimator, ch_client, debug, debug_dir):
    insert_data = []
    
    for method_frame, properties, body in messages:
        try:
            payload_str = body.decode('utf-8') if isinstance(body, bytes) else body
            data = json.loads(payload_str)
            
            # The AMQP body might be the raw JSON payload
            # If it's wrapped like the API example, extract 'payload'
            if isinstance(data, dict) and "payload" in data and isinstance(data["payload"], str):
                payload = json.loads(data["payload"])
            else:
                payload = data
            
            job_id = payload['job']['job_id']
            image_info = payload['image']
            image_id = image_info['image_id']
            image_url = image_info['image_url']
            
            task_info = payload['task']
            output_table = task_info['output_table']
            part_name = task_info['part_name']
            
            model_info = payload['model']
            model_version = model_info['version']
            model_family = model_info.get('family', 'sapiens-1b')
            
            print(f"Processing job {job_id} for image {image_id}")
            
            # Load image
            img = imread_from_url(image_url)
            if img is None:
                print(f"Failed to load image from {image_url}")
                # Acknowledge anyway to avoid poison messages blocking the queue
                channel.basic_ack(delivery_tag=method_frame.delivery_tag)
                continue
                
            # Run inference
            segmentation_map = estimator(img)
            polygons = segmentation_to_polygons(segmentation_map)
            
            h, w = img.shape[:2]
            shape = [float(h), float(w)]
            dtype_str = "float16" # default in schema
            
            # Format polygons as JSON string
            polygons_json = json.dumps(polygons)
            
            # Prepare row for ClickHouse
            row = [
                image_id,
                model_version,
                model_family,
                part_name,
                polygons_json,
                shape,
                dtype_str
            ]
            insert_data.append((row, output_table, method_frame))
            
            if debug:
                os.makedirs(debug_dir, exist_ok=True)
                base_name = f"{image_id}_{job_id}"
                
                cv2.imwrite(os.path.join(debug_dir, f"{base_name}_orig.jpg"), img)
                
                segmentation_image = draw_segmentation_map(segmentation_map)
                combined = cv2.addWeighted(img, 0.5, segmentation_image, 0.7, 0)
                cv2.imwrite(os.path.join(debug_dir, f"{base_name}_seg.jpg"), combined)
                
                with open(os.path.join(debug_dir, f"{base_name}_polygons.json"), 'w') as f:
                    f.write(polygons_json)
                    
        except Exception as e:
            print(f"Error processing message: {e}")
            # Reject to requeue or discard based on your policy (discarding here for safety to avoid loops)
            channel.basic_nack(delivery_tag=method_frame.delivery_tag, requeue=False)
            
    # Group by output_table to do batch inserts
    tables = {}
    for row, table, method_frame in insert_data:
        if table not in tables:
            tables[table] = {'rows': [], 'frames': []}
        tables[table]['rows'].append(row)
        tables[table]['frames'].append(method_frame)
        
    for table, data in tables.items():
        try:
            # ClickHouse columns: image_id, model_version, model_family, part_name, polygons, shape, dtype
            ch_client.insert(table, data['rows'], column_names=['image_id', 'model_version', 'model_family', 'part_name', 'polygons', 'shape', 'dtype'])
            
            # Ack messages only after successful insert
            for frame in data['frames']:
                channel.basic_ack(delivery_tag=frame.delivery_tag)
            print(f"Inserted {len(data['rows'])} rows into {table} and acked messages.")
            
        except Exception as e:
            print(f"ClickHouse insert failed for table {table}: {e}")
            # Nack messages if insert failed so they can be retried
            for frame in data['frames']:
                channel.basic_nack(delivery_tag=frame.delivery_tag, requeue=True)


def main():
    parser = argparse.ArgumentParser(description="Sapiens RabbitMQ Worker")
    parser.add_argument("--config", type=str, default="worker_config.yaml", help="Path to YAML config")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (save images locally)")
    parser.add_argument("--debug-dir", type=str, default="./tmp", help="Directory for debug output")
    args = parser.parse_args()
    
    # ---- CPU thread control ----
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)

    config = load_config(args.config)
    
    # Initialize Sapiens Model
    model_config = config.get("models", {}).get("meta_sapien_segmentation", {})
    model_family = model_config.get("model_family", "sapiens-1b")
    
    if "1b" in model_family:
        model_type = SapiensSegmentationType.SEGMENTATION_1B
    elif "06b" in model_family:
        model_type = SapiensSegmentationType.SEGMENTATION_06B
    else:
        model_type = SapiensSegmentationType.SEGMENTATION_03B
        
    dtype_str = model_config.get("dtype", "float16")
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map.get(dtype_str, torch.float16)

    # ---- FIX: float16 is extremely slow on CPU ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu" and dtype != torch.float32:
        print("CPU detected → forcing float32 (float16 on CPU is very slow)")
        dtype = torch.float32

    print(f"Loading model {model_type} with dtype {dtype}...")
    estimator = SapiensSegmentation(model_type, dtype=dtype, device=device)
    
    # Connect to ClickHouse
    print("Connecting to ClickHouse...")
    ch_client = get_clickhouse_client()
    
    # Connect to RabbitMQ
    print("Connecting to RabbitMQ...")
    connection = get_rabbitmq_connection()
    channel = connection.channel()
    
    # Setup Exchange and Queue
    rmq_config = config.get("rabbitmq", {})
    exchange = rmq_config.get("exchange", "ai.jobs")
    queue_name = rmq_config.get("queue", "ai.inference.jobs")
    routing_keys = rmq_config.get("routing_keys", ["infer.meta_sapien.segmentation.#"])
    
    channel.exchange_declare(exchange=exchange, exchange_type='topic', durable=True)
    channel.queue_declare(queue=queue_name, durable=True)
    
    for rk in routing_keys:
        channel.queue_bind(exchange=exchange, queue=queue_name, routing_key=rk)
        
    channel.basic_qos(prefetch_count=args.batch_size)
    
    print(f"Worker started. Waiting for messages on queue {queue_name}. To exit press CTRL+C")
    
    try:
        while True:
            messages = []
            # Try to get a batch of messages
            for _ in range(args.batch_size):
                method_frame, properties, body = channel.basic_get(queue=queue_name, auto_ack=False)
                if method_frame:
                    messages.append((method_frame, properties, body))
                else:
                    break
            
            if messages:
                process_batch(channel, messages, estimator, ch_client, args.debug, args.debug_dir)
            else:
                time.sleep(1) # Sleep briefly if queue is empty
                
    except KeyboardInterrupt:
        print("Worker stopped by user.")
    finally:
        if connection.is_open:
            connection.close()

if __name__ == "__main__":
    main()
