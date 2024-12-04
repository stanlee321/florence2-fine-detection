import redis
import json
from typing import Union

class RedisClient:
    def __init__(self, host='localhost', port=6379, db=0, password=None):
        self.client = redis.StrictRedis(
            host=host, 
            port=port,
            password=password,
            db=db, 
            decode_responses=True)
    
    def set_value(self, key, value):
        try:
            self.client.set(key, value)
            return True
        except Exception as e:
            print(f"Error setting value in Redis: {e}")
            return False

    def get_value(self, key):
        try:
            value = self.client.get(key)
            return value
        except Exception as e:
            print(f"Error getting value from Redis: {e}")
            return None

# Usage example
if __name__ == "__main__":
    redis_client = RedisClient(host='localhost', port=6379)
    success = redis_client.set_value('test_key', 'test_value')
    if success:
        print("Value set successfully")
    
    value = redis_client.get_value('test_key')
    if value:
        print(f"Retrieved value: {value}")