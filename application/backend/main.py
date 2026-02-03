from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel
import numpy as np
import logging
import time
import os
from nltk import sent_tokenize
import nltk
import json
import torch

# Download NLTK data if not present
try:
    sent_tokenize('test')
except:
    nltk.download('punkt')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('summarization_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Get base path
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger.info(f"Base path: {BASE_PATH}")

# Initialize FastAPI app
app = FastAPI(
    title="AI Summarization API",
    description="Professional-grade summarization API with both extractive and abstractive capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
extractive_model = None
abstractive_pipeline = None

# Load models during startup
@app.on_event("startup")
async def startup_event():
    global extractive_model, abstractive_pipeline
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        # Load extractive model
        logger.info("Loading extractive summarization model...")
        extractive_model = SentenceTransformer(
            os.path.join(BASE_PATH, "modelExtractive"),
            device=device  # Use cpu for broader compatibility
        )
        logger.info("Extractive model loaded successfully")
        
        # Load abstractive model (base model + LoRA adapter)
        logger.info("Loading abstractive summarization model...")
        adapter_path = os.path.join(BASE_PATH, "modelAbstractive")
        
        # 1. Read adapter config to get base model name
        with open(os.path.join(adapter_path, "adapter_config.json"), "r") as f:
            config = json.load(f)
            base_model_name = config["base_model_name_or_path"]
        
        logger.info(f"Loading base model {base_model_name}...")
        
        # 2. Load tokenizer
        tokenizer = T5Tokenizer.from_pretrained(adapter_path)
        
        # 3. Load base model
        base_model = T5ForConditionalGeneration.from_pretrained(
            base_model_name,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 4. Apply LoRA adapter
        logger.info("Applying LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # 5. Create pipeline
        abstractive_pipeline = pipeline(
            task="summarization",
            model=model,
            tokenizer=tokenizer,
            device_map="auto"
        )
        
        logger.info("Abstractive model loaded successfully")
        
        load_time = time.time() - start_time
        logger.info(f"All models loaded in {load_time:.2f} seconds")
        
    except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

# Extractive summarization function
def extractive_summarize(text, num_sentences=3):
    try:
        sentences = sent_tokenize(text, language="english")
        if len(sentences) <= num_sentences:
            return " ".join(sentences)
        
        # Encode sentences
        embeddings = extractive_model.encode(sentences)
        
        # Cluster sentences
        kmeans = KMeans(n_clusters=num_sentences, random_state=42)
        kmeans.fit(embeddings)
        
        # Find closest sentences to cluster centers
        indices = []
        for i in range(num_sentences):
            center = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(embeddings - center, axis=1)
            indices.append(np.argmin(distances))
        
        # Sort indices to maintain original order
        indices.sort()
        
        # Generate summary
        summary = " ".join([sentences[i] for i in indices])
        return summary
    except Exception as e:
        logger.error(f"Error in extractive summarization: {str(e)}")
        raise

# Abstractive summarization function
def abstractive_summarize(text):
    try:
        # Limit input to 512 tokens as per model requirement
        # Rough estimate: 1 token ≈ 4 characters
        if len(text) > 512 * 4:
            text = text[:512 * 4]
        
        result = abstractive_pipeline(text, truncation=True)
        return result[0]["summary_text"]
    except Exception as e:
        logger.error(f"Error in abstractive summarization: {str(e)}")
        raise

# Root endpoint
@app.get("/")
async def root():
    return {"message": "AI Summarization API is running"}

# Health check endpoint
@app.get("/health")
async def health_check():
    if extractive_model and abstractive_pipeline:
        return {"status": "healthy", "models_loaded": True}
    else:
        return {"status": "unhealthy", "models_loaded": False}

# Summarization endpoint
@app.post("/summarize")
async def summarize(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        summarization_type = data.get("type", "extractive")  # extractive or abstractive
        
        # Validate input
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text input is required")
        
        if summarization_type not in ["extractive", "abstractive"]:
            raise HTTPException(status_code=400, detail="Invalid summarization type. Use 'extractive' or 'abstractive'")
        
        # Generate summary
        start_time = time.time()
        if summarization_type == "extractive":
            summary = extractive_summarize(text)
        else:
            summary = abstractive_summarize(text)
        processing_time = time.time() - start_time
        
        logger.info(f"Summarization completed in {processing_time:.2f} seconds")
        
        return {
            "summary": summary,
            "type": summarization_type,
            "processing_time": processing_time,
            "input_length": len(text),
            "output_length": len(summary)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in summarize endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Run the app if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
