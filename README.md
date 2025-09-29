# License Plate OCR API

### Overview

This project provides a FastAPI HTTP endpoint that accepts an image upload, detects a single license plate using a YOLO model, and performs OCR on the cropped plate region using PaddleOCR to return the recognized plate text as JSON.
The detector loads a model file named plate_model.pt at startup, and the OCR engine is initialized with English language and angle classification enabled, while document orientation classification is disabled.
The service performs basic content-type validation, safe image decoding, temporary-file handling, and structured error responses for common failure cases.

### Features

- Single-image POST endpoint at /api/license-plate that returns a JSON object with either a license_plate string or a message describing the outcome.
- YOLO-based license plate detection with highest-confidence box selection, followed by OpenCV preprocessing and PaddleOCR recognition.
- Confidence and geometric filtering of OCR segments using SCORE_LIMIT=0.84 and AREA_LIMIT=0.5 to assemble the final plate string in reading order.


### API

- Endpoint: POST /api/license-plate.
- Request: multipart/form-data with field name file containing an image; content-type must start with image/ and the payload must decode to a valid image.
- Success responses (HTTP 200):
    - {"license_plate": "<TEXT>"} when a plate is detected and recognized.
    - {"message": "No license plate detected"} when no plate is found or assembled.
    - {"message": "Model not sure . Provide with a better image (closer more clear)"} when OCR confidence is below SCORE_LIMIT for the best candidate.
- Error responses:
    - 415 Unsupported Media Type if content-type is missing or not image/*.
    - 400 Bad Request if the uploaded bytes cannot be decoded into an image.
    - 500 Internal Server Error if the server fails to persist the temporary image prior to processing.


### Installation

- Dependencies (from imports): fastapi, ultralytics, paddleocr, opencv-python (or opencv-python-headless), and numpy.
- Example installation:
    - pip install fastapi ultralytics paddleocr opencv-python numpy.
- Ensure the model file plate_model.pt is present in the working directory; the server will raise a FileNotFoundError at startup if it cannot be loaded.


### Run

- App object: app is defined in plates_api.py, suitable for any ASGI server.
- Example command with Uvicorn: uvicorn plates_api:app --host 0.0.0.0 --port 8000.
- Once running, send a multipart/form-data POST request with the image file to the /api/license-plate route on the chosen host and port.


### Request example

- curl -X POST -F "file=@path/to/image.jpg" http://HOST:PORT/api/license-plate.
- Typical success: {"license_plate": "ABC123"}.
- Alternative success messages: {"message": "No license plate detected"} or {"message": "Model not sure . Provide with a better image (closer more clear)"} depending on detection/OCR outcome.


### Configuration

- Model path: model_path = "plate_model.pt" controls which YOLO checkpoint is loaded at startup.
- OCR initialization: PaddleOCR(use_angle_cls=True, lang="en", use_doc_orientation_classify=False) configures English language and angle classification.
- Filtering thresholds: SCORE_LIMIT=0.84 (minimum OCR confidence per fragment) and AREA_LIMIT=0.5 (minimum fragment height as a fraction of cropped plate height) determine which OCR fragments are kept.


### Processing pipeline

- The API reads the uploaded bytes, validates the content-type, decodes the image with OpenCV, and persists it to a temporary file before inference.
- The detector runs the YOLO model, selects the highest-confidence bounding box, and crops the plate region from the original image.
- The crop is preprocessed via grayscale conversion, bilateral filtering, and conversion back to BGR before running PaddleOCR.
- OCR results are filtered by confidence and area ratio, sorted left-to-right by x-coordinate, and concatenated into the final license plate string.


### Response semantics

- Even when no plate is detected or the model is unsure, the API returns HTTP 200 with a message field to signal the outcome instead of using non-2xx codes for these cases.
- Validation and system-level failures return appropriate error codes like 415, 400, or 500 to indicate client-side or server-side errors.



