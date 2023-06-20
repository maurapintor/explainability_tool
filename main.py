from typing import Annotated
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from commit_classifier import ClassifierExplainer
from fastapi.responses import FileResponse

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
templates = Jinja2Templates(directory="templates")
clf = ClassifierExplainer(device="cuda:0")

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/")
async def login(url: Annotated[str, Form()], response_class=HTMLResponse):
    try:
        html = clf.classify_explain(url)
        return HTMLResponse(html)
    except Exception as e:
        return HTMLResponse(str(e))

favicon_path = 'assets/images/favicon.ico'

@app.get('/favicon.ico', include_in_schema=False)
@app.get('/apple-touch-icon.png', include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)

