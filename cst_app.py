from flask import Flask, render_template, request, jsonify
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer,
    BertForMaskedLM,
    BertTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoModelForCausalLM,
    AutoTokenizer
)
import torch
import time
import json
import os

app = Flask(__name__)

class ModelManager:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.training_metrics = {}
        self.load_all_models()

    def load_training_metrics(self, model_path, model_key):
        """Load training metrics from json file"""
        metrics_path = os.path.join(model_path, 'training_metrics.json')
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                self.training_metrics[model_key] = metrics
                print(f"Loaded metrics for {model_key}")
                return metrics
        except Exception as e:
            print(f"Error loading metrics for {model_key}: {str(e)}")
            return None

    def load_model(self, model_path, model_type, dataset):
        """Load a specific model and its tokenizer"""
        try:
            if model_type == "gpt2":
                tokenizer = GPT2Tokenizer.from_pretrained(model_path)
                model = GPT2LMHeadModel.from_pretrained(model_path)
            elif model_type == "bert":
                tokenizer = BertTokenizer.from_pretrained(model_path)
                model = BertForMaskedLM.from_pretrained(model_path)
            elif model_type == "t5":
                tokenizer = T5Tokenizer.from_pretrained(model_path)
                model = T5ForConditionalGeneration.from_pretrained(model_path)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(model_path)

            model_key = f"{model_type}_{dataset}"
            self.models[model_key] = model
            self.tokenizers[model_key] = tokenizer
            
            # Load training metrics
            self.load_training_metrics(model_path, model_key)
            
            print(f"Loaded model: {model_key}")
            return True
        except Exception as e:
            print(f"Error loading model {model_type}_{dataset}: {str(e)}")
            return False

    def load_all_models(self):
    #Load all available models"""
        model_configs = {
        ("models/gpt2_shakespeare", "gpt2", "shakespeare"),
        ("models/gpt2_poetry", "gpt2", "poetry"),
        ("models/gpt2_wiki", "gpt2", "wiki"),
        ("models/gpt2_guanaco", "gpt2", "guanaco"),
        ("models/t5_shakespeare", "t5", "shakespeare"),
        ("models/t5_poetry", "t5", "poetry"),
        ("models/t5_wiki", "t5", "wiki"),
        ("models/t5_guanaco", "t5", "guanaco"),
        ("models/bert_shakespeare", "bert", "shakespeare"),
        ("models/bert_poetry", "bert", "poetry"),
        ("models/bert_wiki", "bert", "wiki"),
        ("models/bert_guanaco", "bert", "guanaco"),
    }



        for model_path, model_type, dataset in model_configs:
            if os.path.exists(model_path):
                self.load_model(model_path, model_type, dataset)

    def get_metrics(self, model_type, dataset):
        """Get training metrics for a specific model"""
        model_key = f"{model_type}_{dataset}"
        return self.training_metrics.get(model_key, {})

    def generate_text(self, prompt, model_type, dataset, max_length=200):
        """Generate text using specified model"""
        model_key = f"{model_type}_{dataset}"
        
        if model_key not in self.models:
            return f"Model {model_key} not found", 0
        
        start_time = time.time()
        model = self.models[model_key]
        tokenizer = self.tokenizers[model_key]
        
        try:
            # Handle different model types
            if model_type == "gpt2":
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                outputs = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            elif model_type == "bert":
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                outputs = model.generate(
                    inputs['input_ids'],
                    max_length=max_length,
                    num_return_sequences=1
                )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            elif model_type == "t5":
                inputs = tokenizer(f"generate story: {prompt}", return_tensors="pt", padding=True, truncation=True)
                outputs = model.generate(
                    inputs['input_ids'],
                    max_length=max_length,
                    num_return_sequences=1
                )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            generation_time = time.time() - start_time
            return generated_text, generation_time
        
        except Exception as e:
            print(f"Error generating text with {model_key}: {str(e)}")
            return f"Error generating text: {str(e)}", 0

# Initialize model manager
model_manager = ModelManager()

@app.route('/')
def home():
    # Get list of available models
    available_models = list(model_manager.models.keys())
    return render_template('index_cst.html', models=available_models)


import base64

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data['prompt']
    model_type = data.get('model_type', 'gpt2')
    dataset = data.get('dataset', 'shakespeare')
    
    # Generate text
    generated_text, generation_time = model_manager.generate_text(
        prompt, 
        model_type, 
        dataset
    )
    
    # Load metrics from the json file
    model_path = f"models/{model_type}_{dataset}"
    metrics_file = os.path.join(model_path, 'training_metrics.json')
    training_metrics = {}
    
    try:
        with open(metrics_file, 'r') as f:
            training_metrics = json.load(f)
    except Exception as e:
        print(f"Error loading metrics: {str(e)}")

    # Get training plot images
    training_plot = None
    epoch_plot = None
    try:
        with open(f"{model_path}/training_metrics.png", 'rb') as f:
            training_plot = f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
    except Exception as e:
        print(f"Error loading training plot: {str(e)}")

    return jsonify({
        'text': generated_text,
        'generation_time': f"{generation_time:.2f}",
        'model': f"{model_type}_{dataset}",
        'training_metrics': training_metrics,
        'training_plot': training_plot,
        'epoch_plot': epoch_plot
    })

if __name__ == '__main__':
    app.run(debug=True)