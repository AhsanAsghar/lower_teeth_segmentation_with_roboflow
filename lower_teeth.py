import cv2
import numpy as np
import os
import sys
import base64
from inference_sdk import InferenceHTTPClient
from inference_sdk.http.errors import HTTPCallErrorError
from roboflow import Roboflow
import sys
import requests


# import some common libraries
import numpy as np
import os, json, cv2, random
from matplotlib import pyplot as plt
from PIL import Image

# Add the utils directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from post_processing_lower import (
    filter_predictions_by_confidence,
    remove_duplicate_predictions,
    correct_predictions,
    draw_predictions
)
        

def detect_lower_teeth():


    image_url = ''
    image = cv2.imread(image_url)

    confidence_threshold = 0.25
    
    # Save the image temporarily
    temp_image_path = "temp_lower_image.jpg"
    cv2.imwrite(temp_image_path, image)


    class DummyFile(object):
        def write(self, x): pass
        def flush(self): pass


    original_stdout = sys.stdout
    sys.stdout = DummyFile()
    rf = Roboflow(api_key="CDxrYtIlfwTupxOwIJDJ")
    project = rf.workspace().project("segmentation-lower-teeth2")
    model = project.version("2").model
    sys.stdout = original_stdout
  
    # Define threshold
    confidence_threshold = 0.18

    response = model.predict(temp_image_path, confidence=18)
    result = response.json()

    

    # Filter predictions by confidence threshold
    filtered_predictions = filter_predictions_by_confidence(result['predictions'], confidence_threshold)

    # First, remove duplicate predictions
    unique_predictions = remove_duplicate_predictions(filtered_predictions, image.shape[1])

    # Then, apply the correction logic
    corrected_predictions = correct_predictions(unique_predictions, image.shape[1])

    # Filter out Predictions
    final_predictions = [pred for pred in corrected_predictions if (41 <= int(pred['class']) <= 48) or (31 <= int(pred['class']) <= 38)]

    # Draw predictions with visible labels
    output_image = draw_predictions(image, final_predictions)


    # Encode the image to base64
    _, buffer = cv2.imencode('.jpg', output_image)
    img_bytes = buffer.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    serializable_predictions = []
    for pred in final_predictions:
        serializable_pred = pred.copy()
        serializable_pred['points'] = [{'x': p['x'], 'y': p['y']} for p in pred['points']]
        serializable_predictions.append(serializable_pred)

  
if __name__ == "__main__":
    # This block will only run when this script is executed directly.
    print("Running directly!")
    detect_lower_teeth()
