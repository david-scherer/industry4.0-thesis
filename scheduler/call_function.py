import requests

def call_nuclio_function(provider: str, data=None):
  """
  Calls a Nuclio function using HTTP.

  Args:
    url: The URL of the Nuclio function.
    data: The data to send to the function.
    headers: The headers to send to the function.

  Returns:
    The response from the function.
  """

  host, port = get_provider_data(provider=provider)

  try:
    response = requests.post(f'{host}:{port}', json=data, headers={"Content-Type": "application/json"})
    #response.raise_for_status()  # Raise an exception for non-200 status codes
    return response.elapsed.total_seconds(), response
  except requests.exceptions.RequestException as e: 
    print(f"Error calling Nuclio function: {e}",flush=True)
    return 0, "Fail"

def get_provider_data(provider):
    if provider == 'edge':
       return 'http://host.docker.internal', 56510
    elif provider == 'private-cloud':
        return 'http://host.docker.internal', 56510
    elif provider == 'public-cloud':
       return 'http://host.docker.internal', 56510
    else:
       return None