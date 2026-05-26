from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os
import shutil
import base64
import docx
import openpyxl

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
INDEX_HTML_PATH = os.path.join(BASE_DIR, "index.html")

class Message(BaseModel):
    message: str

def extract_text_from_docx(file_path: str) -> str:
    try:
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)
    except Exception as e:
        return f"[Error parsing Word file: {str(e)}]"

def extract_text_from_xlsx(file_path: str) -> str:
    try:
        wb = openpyxl.load_workbook(file_path, data_only=True)
        text_content = []
        for sheet in wb.worksheets:
            text_content.append(f"--- Sheet: {sheet.title} ---")
            for row in sheet.iter_rows(values_only=True):
                if any(row):
                    row_str = " | ".join([str(val) if val is not None else "" for val in row])
                    text_content.append(row_str)
        return "\n".join(text_content)
    except Exception as e:
        return f"[Error parsing Excel file: {str(e)}]"

def extract_text_from_txt(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        return f"[Error reading text file: {str(e)}]"

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/chat")
async def chat(msg: Message):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": msg.message}
        ]
    )
    return {"reply": response.choices[0].message.content}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), message: str = Form("")):
    import traceback
    try:
        if not file.filename:
            return JSONResponse(status_code=400, content={"reply": "No file was selected."})
            
        file_extension = os.path.splitext(file.filename)[1].lower()
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        # Save the file locally
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Check if it's an image
        if file_extension in [".png", ".jpg", ".jpeg", ".webp"]:
            # Resolve standard MIME type
            mime_type = "image/jpeg"
            if file_extension == ".png":
                mime_type = "image/png"
            elif file_extension == ".webp":
                mime_type = "image/webp"
                
            # Encode image to base64 for Groq Vision
            with open(file_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            
            prompt = message if message.strip() else "Please describe this image in detail."
            
            response = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{encoded_image}"
                                }
                            }
                        ]
                    }
                ]
            )
            reply = response.choices[0].message.content
            
        else:
            # Parse text contents of document
            extracted_text = ""
            if file_extension == ".docx":
                extracted_text = extract_text_from_docx(file_path)
            elif file_extension in [".xlsx", ".xls"]:
                extracted_text = extract_text_from_xlsx(file_path)
            elif file_extension == ".txt":
                extracted_text = extract_text_from_txt(file_path)
            else:
                extracted_text = "[Unsupported file format. Only images, .docx, .xlsx, and .txt are supported.]"
                
            prompt = message if message.strip() else "Please analyze this file and summarize its content."
            full_prompt = f"User Query: {prompt}\n\n[Extracted File Contents from '{file.filename}']:\n{extracted_text}"
            
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that can analyze files and documents."},
                    {"role": "user", "content": full_prompt}
                ]
            )
            reply = response.choices[0].message.content
            
        return {"reply": reply, "filename": file.filename, "type": file_extension}
        
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"reply": f"An error occurred while processing the file: {str(e)}"})