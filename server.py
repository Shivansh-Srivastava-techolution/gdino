from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import uuid
import os
import shutil
import uvicorn
import traceback
from inference import gdino_image_det

# Import your inference function from your module
from groundingdino.util.inference import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "groundingdino_swint_ogc.pth")

print("Loading model...")
model = load_model(CONFIG_PATH, WEIGHTS_PATH)

app = FastAPI()

# Create directories for temporary input and output images
INPUT_DIR = "temp_inputs"
OUTPUT_DIR = "temp_outputs"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/")
def home():
    return "Server Running"

# Modified endpoint definition - remove trailing slash
@app.post("/infer")
async def infer_image(
    image: UploadFile = File(...),
    prompt: str = Form("medical paper, box")
):
    # Reset file pointer to beginning before reading
    await image.seek(0)
    
    input_filename = os.path.join(INPUT_DIR, f"{uuid.uuid4().hex}_{image.filename}")
    output_filename = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}_processed_{image.filename}")
    
    try:
        # Save uploaded file
        with open(input_filename, "wb") as f:
            shutil.copyfileobj(image.file, f)
        # Process the image
        bboxes = gdino_image_det(model, input_filename, output_filename, prompt)
        bbox_dicts = [{"x": x, "y": y} for x, y in bboxes]
        return {'bboxes': bbox_dicts}
    except Exception as e:
        tb_str = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}\n{tb_str}")
    finally:
        # Clean up the input file after processing
        if os.path.exists(input_filename):
            os.remove(input_filename)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=False)
