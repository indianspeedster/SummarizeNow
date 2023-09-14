from flask import Flask, request, jsonify, render_template
from dataset import DataSet
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import cuda
from flask import Flask, redirect, url_for, request, render_template
from flask import Flask, request, jsonify
device = 'cuda' if cuda.is_available() else 'cpu'

model = T5ForConditionalGeneration.from_pretrained("model_files")
model = model.to(device)
tokenizer = T5Tokenizer.from_pretrained("t5-base")


app = Flask(__name__)

def model_predict(a,b):

    val_dataset_n =  pd.DataFrame({"article":[a], "highlights":[b]})

    val_set = DataSet(val_dataset_n, tokenizer, 512, 120, "article", "highlights")

    val_params = {
      'batch_size': 1,
      'shuffle': False,
      'num_workers': 0
        }
    val_loader = DataLoader(val_set, **val_params)

    model.eval()
    with torch.no_grad():
        for _, data in enumerate(val_loader, 0):
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/summary", methods=["POST"])
def capitalize_text():
    try:
        data = request.get_json()
        text = data["text"]

        # Capitalize the text
        capitalized_text = model_predict(text, text)

        # Return the capitalized text as JSON response
        response_data = {"capitalized_text": text, "Original_text": capitalized_text}
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
