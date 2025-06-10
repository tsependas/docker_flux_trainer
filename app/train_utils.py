import requests
import json
from typing import Optional, Any
import os
import threading
from dotenv import load_dotenv

load_dotenv()

api_url = os.getenv('SERVER_API_URL')
model_id = os.getenv('MODEL_ID')
model_trigger = os.getenv('MODEL_TRIGGER')

def _send_progress_thread(data):
    """Thread function to send progress update"""
    try:
        endpoint = f"{api_url}/api/models/{model_id}/progress"
        response = requests.post(
            endpoint,
            json=data,
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to send training progress: {str(e)}")

def send_progress(
    step: int,
    total_steps: int,
    loss: float,
    learning_rate: float,
    elapsed_time: float,
    status: str = "training"
) -> None:
    """
    Send progress update in a non-blocking way
    
    Args:
        step: Current step
        total_steps: Total steps to complete
        loss: Current loss value
        learning_rate: Current learning rate
        elapsed_time: Time elapsed in seconds
        status: Training status
    """
    try:
        # Calculate progress percentage
        progress = (step / total_steps) * 100
        
        # Calculate estimated time remaining
        if step > 0:
            time_per_step = elapsed_time / step
            remaining_steps = total_steps - step
            eta_seconds = time_per_step * remaining_steps
        else:
            eta_seconds = 0
            
        data = {
            "step": step,
            "total_steps": total_steps,
            "progress": round(progress, 1),
            "loss": loss,
            "learning_rate": learning_rate,
            "elapsed_time": elapsed_time,
            "eta_seconds": eta_seconds,
            "status": status
        }
        
        # Start a thread to send the update
        thread = threading.Thread(target=_send_progress_thread, args=(data,))
        thread.daemon = True  # Thread will be killed when main program exits
        thread.start()
        
    except Exception as e:
        print(f"Failed to queue progress update: {str(e)}")

def get_task() -> Optional[Any]:
    endpoint = f"{api_url}/api/models/{model_id}/task"
    print(f"Endpoint: {endpoint}")

    try:
        response = requests.get(endpoint, headers={'Content-Type': 'application/json'}, timeout=10)
        response.raise_for_status()

        # Make sure 'data' folder exists
        os.makedirs('../input', exist_ok=True)

        # Download photos
        photos = response.json().get('photos', [])
        print(f"Photos: {photos}")

        for photo_url in photos:
            url = photo_url['url']
            filename = url.split("/")[-1]
            photo_response = requests.get(url, timeout=10)
            photo_response.raise_for_status()
            path = os.path.join('../input', filename)
            with open(path, 'wb') as f:
                print(f"Saving photo to {path}")
                f.write(photo_response.content)
            
            # Create a text file with the trigger word for each photo and add it to the input folder (without image extension)
            text_filename = filename.split('.')[0]
            with open(os.path.join('../input', f'{text_filename}.txt'), 'w') as f:
                f.write(model_trigger)
            print(f"Created text file with trigger word for {filename}")

        return response.json()

    except Exception as e:
        print(f"Error fetching task: {e}")
        if 'response' in locals():
            print(f"Response: {response.text}")
        return None

def send_result(result: str) -> bool:
    endpoint = f"{api_url}/api/models/{model_id}/result"
    print(f"Result: {result}")
    try:
        # Check if result is a safetensors file path
        if result.endswith('.safetensors'):
            with open(result, 'rb') as f:
                file_data = {'file': ('model.safetensors', f, 'application/octet-stream')}
                response = requests.post(
                    endpoint,
                    files=file_data
                    # Remove the Content-Type header - requests will set it automatically for multipart/form-data
                )
                print(f"Response: {response.text}")
        else:
            print(f"Result is not a safetensors file: {result}")
        response.raise_for_status()
        return True
    except FileNotFoundError:
        print(f"File not found: {result}")
        return False
    except requests.RequestException as e:
        print(f"Request failed: {str(e)}")
        return False
    except Exception as e:
        print(f"Failed to send result: {str(e)}")
        return False