
import unittest
import json
from unittest.mock import patch, MagicMock
from injest import load_demo_data, KafkaHandler, RedisClient

class TestInjest(unittest.TestCase):

    @patch('os.listdir')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='{"test": "data"}')
    def test_load_demo_data(self, mock_open, mock_listdir):
        mock_listdir.return_value = ['file1.json', 'file2.json', 'output_json_timestamp.json']
        result = load_demo_data('/test/folder')
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], {"test": "data"})
        self.assertEqual(result[1], {"test": "data"})

    @patch('injest.KafkaHandler')
    def test_kafka_produce_message(self, mock_kafka):
        kafka_instance = mock_kafka.return_value
        kafka_instance.produce_message.return_value = None

        kafka_handler = KafkaHandler(bootstrap_servers=['localhost:9092'])
        kafka_handler.produce_message('test-topic', {'key': 'value'})

        kafka_instance.produce_message.assert_called_once_with('test-topic', {'key': 'value'})

    @patch('injest.RedisClient')
    def test_redis_set_value(self, mock_redis):
        redis_instance = mock_redis.return_value
        redis_instance.set_value.return_value = True

        redis_client = RedisClient(host='0.0.0.0', port=6379)
        result = redis_client.set_value('test_key', json.dumps({'test': 'data'}))

        self.assertTrue(result)
        redis_instance.set_value.assert_called_once_with('test_key', '{"test": "data"}')

if __name__ == '__main__':
    unittest.main()
