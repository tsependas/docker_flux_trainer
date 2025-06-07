import requests
import json
from typing import Optional
import os

def send_progress(
    job_id: str,
    step: int,
    total_steps: int,
    loss: float,
    learning_rate: float,
    elapsed_time: float,
    status: str = "training"
) -> bool:
    """
    Send simplified training progress to the API server
    
    Args:
        job_id: Training job name
        step: Current step
        total_steps: Total steps to complete
        loss: Current loss value
        learning_rate: Current learning rate
        elapsed_time: Time elapsed in seconds
        status: Training status
    """
    try:
        api_url = os.getenv('TRAINING_API_URL', 'http://localhost:8000')
        endpoint = f"{api_url}/api/training/progress/{job_id}"
        
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
            
        response = requests.post(
            endpoint,
            json=data,
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        return True
        
    except Exception as e:
        print(f"Failed to send training progress: {str(e)}")
        return False
