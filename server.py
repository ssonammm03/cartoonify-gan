from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch, torchvision, io, base64, os
from PIL import Image
from torchvision import transforms
from model import Generator

# ==============================================================
# 🧩 App Setup
# ==============================================================
app = FastAPI()

# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ==============================================================
# ⚙️ Model Setup
# ==============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "runs/gen_e005.pt"
os.makedirs("outputs", exist_ok=True)

def load_model():
    gen = Generator().to(DEVICE)
    gen.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    gen.eval()
    print("✅ Model loaded successfully!")
    return gen

gen = load_model()

tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)

# ==============================================================
# 🌐 Web Routes
# ==============================================================

# 🏠 Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# 🖼 Cartoonify page (Upload)
@app.get("/cartoonify", response_class=HTMLResponse)
async def cartoonify_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 📷 Camera page
@app.get("/camera", response_class=HTMLResponse)
async def camera_page(request: Request):
    return templates.TemplateResponse("camera.html", {"request": request})

# ==============================================================
# 🔄 API Endpoints
# ==============================================================

# 📤 File Upload → Cartoonify
@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        y = gen(x).cpu().squeeze(0)
    cartoon = denorm(y).permute(1, 2, 0).numpy()

    buf_real = io.BytesIO()
    buf_cartoon = io.BytesIO()
    img.save(buf_real, format="PNG")
    Image.fromarray((cartoon * 255).astype("uint8")).save(buf_cartoon, format="PNG")

    return JSONResponse({
        "original": base64.b64encode(buf_real.getvalue()).decode("utf-8"),
        "result": base64.b64encode(buf_cartoon.getvalue()).decode("utf-8")
    })

# 📷 Webcam Capture → Cartoonify
@app.post("/webcam")
async def webcam_upload(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        y = gen(x).cpu().squeeze(0)
    cartoon = denorm(y).permute(1, 2, 0).numpy()

    buf_real = io.BytesIO()
    buf_cartoon = io.BytesIO()
    img.save(buf_real, format="PNG")
    Image.fromarray((cartoon * 255).astype("uint8")).save(buf_cartoon, format="PNG")

    return JSONResponse({
        "original": base64.b64encode(buf_real.getvalue()).decode("utf-8"),
        "result": base64.b64encode(buf_cartoon.getvalue()).decode("utf-8")
    })



