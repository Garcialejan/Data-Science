import pickle
from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()
templates = Jinja2Templates(directory="templates")

path_model = "./models/ridge.pkl"
path_scaler = "./models/scaler.pkl"

try:
    with open(path_model, "rb") as file:
        model = pickle.load(file)
    
    with open(path_scaler, "rb") as file:
        scaler = pickle.load(file)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")
    
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
@app.get("/", response_class=HTMLResponse, status_code= 200)
async def index(request: Request):
    return templates.TemplateResponse("index.html",
                                      {"request":request})

# Build the prediction
@app.get("/predictdata", response_class=HTMLResponse)
@app.post("/predictdata", response_class=HTMLResponse)
async def predict_datapoint(request:Request,
                            Temperature: float = Form(...),
                            RH: float = Form(...),
                            Ws: float = Form(...),
                            Rain: float = Form(...),
                            FFMC: float = Form(...),
                            DMC: float = Form(...),
                            ISI: float = Form(...),
                            Classes: float = Form(...),
                            Region: float = Form(...)):
    if request.method == "POST":
        # Crea un diccionario con los datos de entrada
        input_data = {
            "Temperature": Temperature,
            "RH": RH,
            "Ws": Ws,
            "Rain": Rain,
            "FFMC": FFMC,
            "DMC": DMC,
            "ISI": ISI,
            "Classes": Classes,
            "Region": Region
        }
        input_list = [input_data[key] for key in input_data]
        new_data_scaled=scaler.transform([input_list])
        result=model.predict(new_data_scaled)[0]
        
        return templates.TemplateResponse('home.html',
                                          {"request": request, "result": result})
    else:
        return templates.TemplateResponse('home.html',
                                          {"request": request})