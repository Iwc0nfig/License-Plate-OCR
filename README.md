

## **License Plate Recognition API**

A FastAPI-based application that detects and recognizes license plates from images using YOLO (for object detection) and PaddleOCR (for text recognition). The service is containerized for easy deployment and optimized for CPU usage.

## **📋 Prerequisites & Requirements**

Before running the application, ensure you have the following:

* **Operating System:** Linux (Recommended) or Windows.

* **Model File:** The custom YOLO model plate_model.pt **must** be present in the root directory. It is not included in standard libraries.

* **Internet Access:** Required on the first run to download default PaddleOCR models (\~20MB).

---

**🚀 Installation & Usage**

You can run the API using **Docker** (recommended) or a **Local Environment**.

### **Option A: Docker (Recommended)**

The Docker setup includes CPU-optimized versions of PyTorch and PaddlePaddle to keep image size manageable.

1. **Build and Run**:
```bash
docker-compose up \--build
```

   This will start the service on port 8001.

### **Option B: Local Environment**

If running without Docker, follow these steps to set up the environment manually.

1. **Install System Dependencies** (Linux/Ubuntu): OpenCV requires specific GL libraries on headless Linux servers.

```bash
sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6 \-y
```

2. **Set up Python Environment**: It is recommended to use Conda with Python 3.10.

```bash
conda create \-n plates-env python=3.10 \-y
conda activate plates-env
```

3. **Install Python Packages**:
   First, ensure paddlepaddle is installed (required by PaddleOCR).
```bash
pip install paddlepaddle
```

   Then install the remaining dependencies:
```bash
pip install \-r requirements.txt
```

4. **Start the Server**:
```bash
uvicorn plates\_api:app \--host 0.0.0.0 \--port 8001
```

---

**🔌 API Documentation**

**Base URL:** http://<SERVER_IP>:8001

### **Detect License Plate**

* **Endpoint:** POST /api/license-plate

* **Content-Type:** application/json

* **Body:** A JSON object containing the Base64 encoded image string.

#### **Request Example**

```json
{
"image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
}
```

Note: The API handles raw Base64 strings or those with the data:image prefix.

#### **Responses**

1. **Success (Plate Detected)**
```json
{
"license\_plate": "ABC1234"
}
```

2. **No Plate Detected**
```json
{
"message": "No license plate detected"
}
```

3. **Low Confidence (Blurry/Far Image)**
```json
{
"message": "Model not sure . Provide with a better image (closer more clear)"
}
```

---

**📂 Project Structure**

Ensure your deployment directory looks like this:

```plaintext
.
├── plates\_api.py        \# Main application code \[cite: 15\]
├── requirements.txt     \# Python dependencies \[cite: 17\]
├── plate\_model.pt       \# Custom YOLO model (CRITICAL)
├── Dockerfile           \# Container definition
├── docker-compose.yml   \# Orchestration config
└── .dockerignore        \# Build exclusion rules
```

---

**🛠️ Troubleshooting**

* **ModuleNotFoundError**: Ensure paddlepaddle is installed (pip install paddlepaddle).

* **FileNotFoundError**: Verify that plate_model.pt is located in the root directory where the command is run.

* **ImportError: libGL.so.1**: Missing system libraries. Run the apt-get install command listed in the **Installation** section.

* **Server Errors (500)**: Usually indicates a failure to process the temporary file or a model runtime error.

---

**Would you like me to create a shell script to automate the local installation steps for you?**
