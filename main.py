from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel

from PIL import Image
import torch
import torchvision.transforms as transforms

from helper_lib.model import get_model


app = FastAPI()

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. It tells the story of Edmond Dantes who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)
embedding_model = EmbeddingModel()

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

classifier = get_model("EnhancedCNN")

checkpoint = torch.load(
    "checkpoints/model_epoch_009.pth",
    map_location="cpu"
)

classifier.load_state_dict(checkpoint["model_state_dict"])
classifier.eval()

image_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(
        request.start_word,
        request.length
    )
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    embedding = embedding_model.calculate_embedding(request.word)

    return {
        "word": request.word,
        "embedding": embedding
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    image = image_transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = classifier(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    return {
        "predicted_class": CLASSES[predicted_class.item()],
        "confidence": round(confidence.item(), 4)
    }