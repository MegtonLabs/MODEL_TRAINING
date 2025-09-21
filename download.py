
import os
from transformers import BertTokenizer, BertModel

MODEL_NAME_FROM_HUB = "bert-base-uncased"
LOCAL_SAVE_PATH = "./local-bert-base-uncased"  # <-- different name

print("--- Starting Model and Tokenizer Download ---")

os.makedirs(LOCAL_SAVE_PATH, exist_ok=True)

# --- Download and Save the Base Model ---
try:
    print(f"Downloading base model '{MODEL_NAME_FROM_HUB}' from Hugging Face Hub...")
    model = BertModel.from_pretrained(MODEL_NAME_FROM_HUB)

    print(f"Saving model to local directory: '{LOCAL_SAVE_PATH}'")
    model.save_pretrained(LOCAL_SAVE_PATH)
    print("✅ Base model saved successfully.")
except Exception as e:
    print(f"❌ Error while downloading or saving the base model: {e}")


# --- Download and Save the Tokenizer ---
try:
    print(f"\nDownloading tokenizer for '{MODEL_NAME_FROM_HUB}'...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME_FROM_HUB)

    print(f"Saving tokenizer to local directory: '{LOCAL_SAVE_PATH}'")
    tokenizer.save_pretrained(LOCAL_SAVE_PATH)
    print("✅ Tokenizer saved successfully.")
except Exception as e:
    print(f"❌ Error while downloading or saving the tokenizer: {e}")

print("\n--- Download process finished. ---")
