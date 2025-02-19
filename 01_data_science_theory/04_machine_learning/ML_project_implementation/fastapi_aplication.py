import pickle
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()
templates = Jinja2Templates(directory="templates")

path_model = "./models/ridge.pkl"
path_scaler = "./models/scaler.pkl"

with open(path_model, "rb") as file:
    model = pickle.load(file)
    
with open(path_scaler, "rb") as file:
    scaler = pickle.load(file)
    
class InputData(BaseModel):
    Temperature: float
    RH: float
    Ws: float 
    Rain: float 
    FFMC: float
    DMC: float 
    ISI: float 
    Classes: float
    Region: float 
    
# Home page
@app.get("/", response_model=HTMLResponse, status_code= 200)
async def index(request: Request):
    return templates.TemplateResponse("home.html",
                                      {"request":request})

# Build the prediction
@app.get("/predictdata", response_class=HTMLResponse)
@app.post("/predictdata", response_class=HTMLResponse)
async def predict_datapoint(request:Request):
    if request.method == "POST":
        
