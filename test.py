# File: predict.py

import torch
import sys
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

# --- Configuration ---
# Path to the directory where the fine-tuned model is saved
MODEL_PATH = './model_save/'

# The same MAX_LEN that you used during training
MAX_LEN = 160

# --- Main Prediction Function ---
def predict_spam(text):
    """
    Loads the fine-tuned model and tokenizer to predict if a text is spam or ham.
    """
    # Set up the device (GPU or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU for prediction.")

    # --- Load Tokenizer and Model ---
    try:
        print(f"Loading tokenizer from: {MODEL_PATH}")
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        
        print(f"Loading model from: {MODEL_PATH}")
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        print("Please make sure the 'model_save' directory exists and contains the fine-tuned model.")
        return

    # Move the model to the selected device
    model.to(device)

    # Set the model to evaluation mode. This turns off layers like Dropout.
    model.eval()

    # --- Preprocess the Input Text ---
    # The input text must be processed in the exact same way as the training data
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',  # Return PyTorch tensors
    )

    # Move tensors to the same device as the model
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)

    # --- Make a Prediction ---
    # Tell PyTorch not to calculate gradients, as we are only doing inference
    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
    
    # The output is a tuple where the first element contains the logits
    logits = outputs.logits

    # --- Process the Output ---
    # Apply softmax to the logits to get probabilities
    probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
    
    # Get the prediction (0 for 'ham', 1 for 'spam')
    prediction_index = torch.argmax(logits, dim=1).item()
    
    # Map the index to the label
    labels = ['Ham', 'Spam']
    prediction = labels[prediction_index]

    # Return a dictionary with the results
    return {
        'prediction': prediction,
        'confidence': {
            'ham': f"{probabilities[0]*100:.2f}%",
            'spam': f"{probabilities[1]*100:.2f}%"
        }
    }

# --- Command-Line Interface ---
if __name__ == '__main__':
    # Check if a text argument was provided
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"<your text message here>\"")
    else:
        # Join all arguments after the script name to form the input text
        input_text = " ".join(sys.argv[1:])
        
        print(f"\nAnalyzing text: \"{input_text}\"")
        print("-" * 30)
        
        result = predict_spam(input_text)
        
        if result:
            print(f"\nPrediction: {result['prediction']}")
            print(f"Confidence:")
            print(f"  - Ham:  {result['confidence']['ham']}")
            print(f"  - Spam: {result['confidence']['spam']}")
            print("-" * 30)