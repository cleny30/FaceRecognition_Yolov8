from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import zipfile
import tempfile
import encode_faces
import recognize_faces
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# Directory to save uploaded files temporarily
UPLOAD_DIR = "uploads"
ENCODINGS_PATH = 'class'
DATASET_PATH = 'dataset'

encode_file = 'class/encodings.pickle'
detection_method = "hog"
app.add_middleware(
    CORSMiddleware,
    # Replace with your frontend domains
    allow_origins=["http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["POST", "DELETE"],  # Allow only specific methods
    allow_headers=["*"],  # Allow all headers
)

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

async def train_dataset():
    print("[INFO] Starting training on the extracted dataset...")
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        # Chạy encode_faces.encode_faces trong executor
        message = await loop.run_in_executor(pool, encode_faces.encode_faces, DATASET_PATH, encode_file)
        # Chạy clear_directory trong executor
        await loop.run_in_executor(pool, clear_directory, DATASET_PATH)
    print("[INFO] Training completed:", message)

@app.post("/upload")
async def upload_zip(file: UploadFile = File(...)):
    try:
        # Check if the uploaded file is a ZIP file
        if not file.filename.endswith('.zip'):
            return JSONResponse(content={"status": "error", "message": "Uploaded file is not a ZIP file."}, status_code=400)

        # Create a temporary directory to store the uploaded ZIP file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_zip_path = os.path.join(temp_dir, file.filename)

            # Save the uploaded file to the temporary directory
            with open(temp_zip_path, "wb") as temp_file:
                temp_file.write(await file.read())

            # Extract the ZIP file to the extraction path
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATASET_PATH)

        # Run training dataset in the background
        asyncio.create_task(train_dataset())
        return JSONResponse(content={"status": "success", "message": "File uploaded and processing started."}, status_code=200)

    except zipfile.BadZipFile:
        return JSONResponse(content={"status": "error", "message": "Invalid ZIP file."}, status_code=400)

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


@app.post("/detect")
async def detect_faces(file: UploadFile = File(...)):
    """
    Endpoint to detect and recognize faces in an uploaded image.
    """
    try:
        # Save the uploaded file temporarily
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        classifier_loaded = os.path.exists(encode_file)
        if not classifier_loaded:
            return JSONResponse(content={"status": "error", "message": "Please upload dataset before detecting!"}, status_code=500)
        # Call the face recognition function
        results = recognize_faces.recognize_faces_in_image(
            encode_file,
            file_path
        )

        # Return the results
        return JSONResponse(results)

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

    finally:
        # Clean up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)


@app.delete("/delete")
def clear_directories():
    try:
        # Clear files in OUTPUT_PATH
        clear_directory(DATASET_PATH)
        # Clear files in CLASS_PATH
        clear_directory(ENCODINGS_PATH)
        return JSONResponse(content={'message': 'Directories cleared successfully'}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


# Run the FastAPI app
if __name__ == "__main__":
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(ENCODINGS_PATH, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=5001)
