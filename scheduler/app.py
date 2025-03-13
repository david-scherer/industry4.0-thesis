from flask import Flask, request, jsonify
import os
import logging
import time
from datetime import datetime

from scheduler.dql_scheduler import choose_instance
from scheduler.call_function import call_nuclio_function
from scheduler.entities.instance_mock import InstanceMock
from scheduler.entities.serverless_funcion import Function
from scheduler.utils.config_reader import ConfigReader
from scheduler.training.environment import ServerlessSchedulingEnv

from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core import DEFAULT_MODULE_ID

config = ConfigReader()
model_path = config.read_config("General", "MODEL_PATH")
rl_module: RLModule = RLModule.from_checkpoint(
        os.path.join(
            model_path,
            "learner_group",
            "learner",
            "rl_module",
            DEFAULT_MODULE_ID,
        )
    )
instances = [InstanceMock('edge', 30, 0.95, (5, 15)),
             InstanceMock('private-cloud', 30, 0.99, (16, 25)),
             InstanceMock('public-cloud', 30, 0.9995, (25, 40))]
functions = [Function(['critical'], 'test1'), 
             Function(['critical'], 'test2'), 
             Function(['critical'], 'coldstart')]
env = ServerlessSchedulingEnv(env_config={"providers": instances, "functions": functions})

app = Flask(__name__)


log_file = "scheduler-logs/scheduler.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)

def log_experiment(arrival_time, scheduling_time, nuclio_call_duration, float_values):
  with open(log_file, "a") as f:  # Open the file in append mode

    log_message = f"Arrival {arrival_time}, Scheduling {scheduling_time}, Duration Call {nuclio_call_duration}, with Values {float_values}\n"
    f.write(log_message)
      

def extract_float_values(payload):
  """
  Extracts float values from the payload's temperature field and returns them as a list.

  Args:
    payload: The JSON payload received from Orion Context Broker.

  Returns:
    A list of float values, preserving the order from the payload, or None if 
    the structure is unexpected or no values are found.
  """
  try:
    data_items = payload.get('data')
    if not data_items or not isinstance(data_items, list):
      return None

    # Assuming there's only one item in 'data' list as per your example.
    # You might need to adjust this if your payload can have multiple items.
    temperature_data = data_items[0].get('temperature')
    if not temperature_data or temperature_data.get('type') != 'StructuredValue':
      return None
    
    value_list = temperature_data.get('value')
    if not value_list or not isinstance(value_list, list):
      return None

    float_values = []
    for item in value_list:
      if item.get('type') == 'Property' and 'value' in item:
        float_values.append(item['value'])

    return float_values

  except (KeyError, AttributeError) as e:
    print(f"Error extracting values: {e}",flush=True)
    return None

@app.route('/notify', methods=['POST'])
def handle_notify():
    """Handles notifications from Orion Context Broker."""
    try:
        arrival_time = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        data = request.get_json()

        float_values = extract_float_values(data)

        if float_values:
            action = choose_instance(instances=instances, float_values=float_values, rl_module=rl_module, env=env)
            scheduling_time = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            if action == 0:
                provider = 'edge'
            else:
                provider = 'private-cloud'

            payload = {
              "sensor_id": "your_sensor_id",
              "sensor_values": float_values
            }

            nuclio_call_duration, res = call_nuclio_function(provider=provider, data=payload)
            print(res,flush=True)
            try:
              log_experiment(arrival_time=arrival_time, scheduling_time=scheduling_time, nuclio_call_duration=nuclio_call_duration, float_values=float_values)
            except Exception as ex:
              print(f"amk logging goht net: {e}",flush=True)
        else:
            print("Could not extract float values.",flush=True)

        return jsonify({'message': 'Notification received'}), 200

    except Exception as e:
        print(f"Error handling notification: {e}",flush=True)
        return jsonify({'error': 'Failed to process notification'}), 500
    
@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
