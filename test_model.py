from transformers import AutoModelForSequenceClassification

try:
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert_hate_speech",
        local_files_only=True
    )
    print("✅ Model loaded successfully!")
    print(f"Model type: {type(model).__name__}")
except Exception as e:
    print(f"❌ Load failed: {type(e).__name__}: {e}")