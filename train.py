import pandas as pd
import torch
import numpy as np
import time
import datetime
import os
import glob  # New import to find all files in a directory
from torch.utils.data import TensorDataset, DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW

# --- Configuration & Hyperparameters ---
MODEL_NAME = './local-bert-base-uncased'
# This now points to the folder containing all your dataset CSVs
DATASET_DIRECTORY = './datasets/'
MAX_LEN = 160
BATCH_SIZE = 16
# Consider starting with 3 epochs when using a larger, combined dataset
EPOCHS = 3
LEARNING_RATE = 2e-5
ADAM_EPSILON = 1e-8
# ----------------------------------------

# --- Device Setup ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# --- NEW: Multi-Dataset Loading and Combining Function ---
def load_and_standardize_datasets(path):
    """
    Loads all CSV files from a directory, standardizes their columns and labels,
    combines them, removes duplicates, and shuffles the result.
    """
    all_files = glob.glob(os.path.join(path, "*.csv"))
    if not all_files:
        print(f"Error: No CSV files were found in the directory '{path}'.")
        print("Please make sure your CSV files are inside the 'datasets' folder.")
        exit()

    print(f"Found {len(all_files)} dataset files to process...")

    df_list = []
    for filename in all_files:
        try:
            # Attempt to load with common encodings
            df = pd.read_csv(filename, encoding='latin-1')

            # --- Standardize Column Names ---
            # Add any other column names you encounter in your datasets here
            column_rename_map = {
                'v1': 'label', 'v2': 'text',
                'Category': 'label', 'Message': 'text',
                'CLASS': 'label', 'CONTENT': 'text'
            }
            df = df.rename(columns=lambda c: c.strip().lower()).rename(columns=column_rename_map)

            if 'label' not in df.columns or 'text' not in df.columns:
                print(f"--> Skipping file: '{os.path.basename(filename)}'. Could not find required 'label' and 'text' columns.")
                continue

            # Keep only the standardized columns
            df = df[['label', 'text']]

            # --- Standardize Labels ---
            # Convert labels to string, lowercased, and then map to 0s and 1s
            label_map = {
                'ham': 0, 'spam': 1,
                '0': 0, '1': 1,
                'normal': 0,
                'legitimate': 0
            }
            df['label'] = df['label'].astype(str).str.lower().map(label_map)

            # Drop rows where label or text is missing, or label couldn't be mapped
            df.dropna(inplace=True)
            df['label'] = df['label'].astype(int)

            df_list.append(df)
            print(f"--> Successfully loaded and processed '{os.path.basename(filename)}', adding {len(df)} rows.")

        except Exception as e:
            print(f"--> Error processing file '{os.path.basename(filename)}': {e}")

    if not df_list:
        print("\nError: No data could be loaded from any of the files. Exiting.")
        exit()

    # --- Combine, Deduplicate, and Shuffle ---
    master_df = pd.concat(df_list, ignore_index=True)
    print(f"\nTotal combined rows: {len(master_df):,}")

    master_df.drop_duplicates(subset=['text'], inplace=True)
    print(f"Rows after removing duplicate text entries: {len(master_df):,}")

    master_df = master_df.sample(frac=1).reset_index(drop=True)
    print("Final dataset shuffled and ready for training.")

    return master_df

# --- Helper Functions (Unchanged) ---
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# --- Main Execution ---

# 1. Load and Prepare Data using the new multi-dataset function
print("\n--- Step 1: Loading and Preparing Data ---")
df = load_and_standardize_datasets(DATASET_DIRECTORY)

sentences = df.text.values
labels = df.label.values

# 2. Tokenization
print(f"\n--- Step 2: Tokenizing Data ---")
print(f"Loading BERT tokenizer from local path: '{MODEL_NAME}'...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

input_ids = []
attention_masks = []

for sent in sentences:
    encoded_dict = tokenizer.encode_plus(
                        sent,
                        add_special_tokens=True,
                        max_length=MAX_LEN,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt',
                   )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# 3. Create Datasets and DataLoaders
print("\n--- Step 3: Creating DataLoaders ---")
dataset = TensorDataset(input_ids, attention_masks, labels)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print(f'{train_size:,} training samples')
print(f'{val_size:,} validation samples')

train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=BATCH_SIZE)
validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=BATCH_SIZE)

# 4. Load Pre-trained Model
print(f"\n--- Step 4: Loading Pre-trained Model ---")
print(f"Loading pre-trained BERT model from local path: '{MODEL_NAME}'...")
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
)
model.to(device)

# 5. Setup Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=ADAM_EPSILON)
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 6. Training Loop
print("\n--- Step 5: Starting Training ---")
total_t0 = time.time()
training_stats = []

for epoch_i in range(0, EPOCHS):
    print(f"\n======== Epoch {epoch_i + 1} / {EPOCHS} ========")
    print('Training...')
    t0 = time.time()
    total_train_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print(f'  Batch {step:>5,}  of  {len(train_dataloader):>5,}.    Elapsed: {elapsed}.')

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()
        result = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels, return_dict=True)
        loss = result.loss
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    training_time = format_time(time.time() - t0)
    print(f"\n  Average training loss: {avg_train_loss:.2f}")
    print(f"  Training epoch took: {training_time}")

    print("\nRunning Validation...")
    t0 = time.time()
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0

    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            result = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels, return_dict=True)

        loss = result.loss
        logits = result.logits
        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print(f"  Accuracy: {avg_val_accuracy:.2f}")
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    validation_time = format_time(time.time() - t0)
    print(f"  Validation Loss: {avg_val_loss:.2f}")
    print(f"  Validation took: {validation_time}")

    training_stats.append({
        'epoch': epoch_i + 1,
        'Training Loss': avg_train_loss,
        'Valid. Loss': avg_val_loss,
        'Valid. Accur.': avg_val_accuracy,
        'Training Time': training_time,
        'Validation Time': validation_time
    })

print("\nTraining complete!")
print(f"Total training took {format_time(time.time()-total_t0)} (h:mm:ss)")

# 7. Save the final model
print("\n--- Step 6: Saving Final Model ---")
output_dir = './model_save/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Saving final model to {output_dir}")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("Save complete.")