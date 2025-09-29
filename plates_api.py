import cv2
import numpy as np
from ultralytics import YOLO
from sys import exit
from paddleocr import PaddleOCR
import tempfile
import os

from fastapi import FastAPI, File, UploadFile , HTTPException
from fastapi.responses import JSONResponse

model_path = "plate_model.pt"
SCORE_LIMIT = 0.84
AREA_LIMIT = 0.5

try:
    model = YOLO(model_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Model file {model_path} not found. Please ensure the model file is in the correct directory.")
except Exception as e:
    raise RuntimeError(f"An error occurred while loading the model: {e}")


try:
    ocr = PaddleOCR(use_angle_cls=True, lang="en",use_doc_orientation_classify=False) 
except Exception as e:
    raise RuntimeError(f"An error occurred while initializing PaddleOCR: {e}")


def detect_license_plate(image_path: str) -> str:
    """
    Detects license plates in the given image using a pre-trained YOLO model.

    Args:
        image (np.ndarray): The input image in which to detect license plates.

    Returns:
        np.ndarray: The image with detected license plates highlighted.
    """

    image = cv2.imread(image_path)
    try:
        results = model(image)
    
    except Exception as e:
        raise RuntimeError(f"An error occurred during model inference: {e}")

    boxes = results[0].boxes.xyxy.tolist()  # Get bounding boxes in integer format
    if not boxes:
        return ""

    box_probs = results[0].boxes.conf.tolist()  # Get confidence scores

    # Find the box with the highest confidence score
    box_highest_prob = max(box_probs)
    index = box_probs.index(box_highest_prob)
    

    x1, y1, x2, y2 = map(int, boxes[index])
    

    plate = image[y1:y2, x1:x2].copy()
    

    # Preprocess the plate image
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 75, 75)
    
    prep = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    
      # recognition-only
    res = ocr.ocr(prep)

    rec_texts = res[0]['rec_texts']
    rec_scores = res[0]['rec_scores']
    rec_boxes = res[0]['rec_boxes']
    
    license_plate_list = []
    for text, score, box in zip(rec_texts, rec_scores, rec_boxes):
        if score > SCORE_LIMIT and len(text) < 10:
            x1, y1, _, y2 = map(int, box)  
            h =  y2 - y1
            area_percentage  = h / plate.shape[0]
            if area_percentage > AREA_LIMIT:
                license_plate_list.append((text,x1))

    
    
    license_plate_list.sort(key=lambda x:x[1])
    license_plate = ''.join(text for text,_ in license_plate_list)
    return license_plate , min(rec_scores ) if rec_scores else 0 

app = FastAPI()

@app.post("/api/license-plate")
async def api_license_plate(file:UploadFile=File(...)):
    # Basic content-type validation
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported media type; expected image/*")
    
    data = await file.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        ok = cv2.imwrite(tmp.name, image)
        temp_path = tmp.name

    try:
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to write image to temporary file")
        license_plate , scores = detect_license_plate(temp_path)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            raise f"Failed to delete temporary file with path: {temp_path}"
        
    if not license_plate :
        if  SCORE_LIMIT>scores:
            return JSONResponse(content={"message": "Model not sure . Provide with a better image (closer more clear)"}, status_code=200)
        return JSONResponse(content={"message": "No license plate detected"}, status_code=200)
        
            
    return JSONResponse(content={"license_plate": license_plate}, status_code=200)

