#pip install SentencePiece --trusted-host pypi.org --trusted-host files.pythonhosted.org --proxy http://FSPHSLV3062:hsl%401234@172.21.5.155:3128
#  pip install Safetensors --trusted-host pypi.org --trusted-host files.pythonhosted.org --proxy http://FSPHSLV3062:hsl%401234@172.21.5.155:3128
# pip install torch torchvision torchaudio --trusted-host pypi.org --trusted-host files.pythonhosted.org --proxy http://FSPHSLV3062:hsl%401234@172.21.5.155:3128 --index-url https://download.pytorch.org/whl/cu121
import csv
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load model and tokenizer from local path
model_path = "./flan-t5-large"  # Replace with your correct path

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(
    model_path,
    use_safetensors=True,         # Since you're using model.safetensors
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load your CSV (replace column name if needed)
df = pd.read_csv("emails.csv")  # Make sure 'Email_message' column exists

def classify_and_extract(index,email_text):
    print(f'Processing {index}')

    prompt = (
        "Classify the following email into one of the following categories:\n"
        "- Bank Account Query\n"
        "- Demat Query (also consider any email about trading, holding, Client Master List, CML, stock, share market, or related terms as part of this category)\n"
        "- Bank Account Closure Query\n"
        "If it doesn't match any category, respond with: Unknown\n\n"
        "Important:\n"
        "- Ignore lines like 'Classification: Internal' or 'Classification: public'.\n"
        "- Do NOT classify into any category other than the three provided above.\n\n"
        f"Email:\n{email_text}\n\n"
        "Category:"
    )

    # prompt = (
    #     "You are a helpful assistant. Read the email carefully.\n"
    #     "Task:\n"
    #     "1. Classify into: Bank Account Query, Demat Query, Bank Account Closure Query.\n"
    #     "Rules:\n"
    #     "- Use 'Unknown' if not classified into given category.\n"
    #     f"Email: {email_text}"
    # )

    # Tokenize and move input to same device as model
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    # Generate output
    output = model.generate(**inputs, max_new_tokens=128)

    # Decode the output
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded

def parse_output(text):
    """Parse the output into a structured dict"""
    result = {"Category": "", "Account Number": "", "Date": "", "Issue Summary": ""}
    for line in text.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            if key.strip() in result:
                result[key.strip()] = val.strip()
    return result

# Run classification and extraction
# parsed_results = df["Email_message"].apply(lambda x: parse_output(classify_and_extract(x)))
#df = df.head(100)
parsed_results = df.apply(lambda x: classify_and_extract(x.name,x["Email_message"]),axis=1)

# Convert dict results to DataFrame and merge
# parsed_df = pd.DataFrame(parsed_results.tolist())
# df_updated = pd.concat([df, parsed_df], axis=1)

df['new_category'] = parsed_results
# Save to CSV
df.to_csv("emails_classified.csv", index=False,quoting=csv.QUOTE_ALL)

print("Classification completed and saved to emails_classified.csv")
