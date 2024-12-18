from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import time
import json
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from datetime import datetime
import re
import random
import warnings
warnings.filterwarnings("ignore")

def simple_tokenize(text):
    """Simple tokenization that splits on spaces and punctuation"""
    # Remove extra spaces
    text = ' '.join(text.split())
    # Split on spaces and punctuation
    tokens = [token.strip('.,!?()[]{}":;') for token in text.lower().split()]
    return [token for token in tokens if token]

def calculate_bleu(reference, candidate):
    """Calculate BLEU score using simple tokenization"""
    try:
        # Use simple tokenization
        ref_tokens = simple_tokenize(reference)
        cand_tokens = simple_tokenize(candidate)
        
        # If either is empty, return 0
        if not ref_tokens or not cand_tokens:
            return 0.0
            
        # Calculate n-gram matches
        matches = sum(1 for token in cand_tokens if token in ref_tokens)
        
        # Simple BLEU calculation
        precision = matches / len(cand_tokens) if len(cand_tokens) > 0 else 0
        return precision
    except:
        return 0.0

def evaluate_model_bleu(model, tokenizer, eval_texts, num_samples=10):
    """Evaluate model using BLEU score on a set of texts"""
    bleu_scores = []
    
    for text in random.sample(eval_texts, min(num_samples, len(eval_texts))):
        try:
            # Take first half as prompt and second half as reference
            split_point = len(text) // 2
            prompt = text[:split_point]
            reference = text[split_point:]
            
            # Generate continuation
            inputs = tokenizer(
                prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=min(len(inputs['input_ids'][0]) + 50, 200),
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )
            
            generated_text = tokenizer.decode(
                outputs[0][len(inputs['input_ids'][0]):], 
                skip_special_tokens=True
            )
            
            # Calculate BLEU score
            bleu_score = calculate_bleu(reference, generated_text)
            bleu_scores.append(bleu_score)
            
        except Exception as e:
            print(f"Warning: Error in BLEU calculation: {str(e)}")
            continue
    
    return np.mean(bleu_scores) if bleu_scores else 0.0

def preprocess_text(text):
    """Clean and preprocess the text data"""
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters while keeping basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-:;]', '', text)
    # Ensure proper spacing after punctuation
    text = re.sub(r'([.,!?])', r'\1 ', text)
    # Remove multiple punctuation
    text = re.sub(r'([.,!?])\1+', r'\1', text)
    # Remove extra spaces again after cleaning
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def create_training_chunks(text, chunk_size, overlap_size):
    """Create chunks with overlap for better coherence"""
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size]
        if len(chunk) >= chunk_size // 2:  # Only keep chunks that are at least half the desired size
            chunks.append(chunk)
        start += chunk_size - overlap_size
    return chunks

class CustomTrainer(Trainer):
    def __init__(self, *args, eval_texts=None, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_loss_history = []
        self.eval_texts = eval_texts
        self.tokenizer = tokenizer
        self.bleu_history = []
    
    def log(self, logs, start_time=None, **kwargs):
        super().log(logs)
        if 'loss' in logs:
            self.training_loss_history.append(logs['loss'])
            print(f"Current loss: {logs['loss']:.4f}")
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if self.eval_texts and self.tokenizer:
            try:
                bleu_score = evaluate_model_bleu(self.model, self.tokenizer, self.eval_texts)
                self.bleu_history.append(bleu_score)
                print(f"Current BLEU score: {bleu_score:.4f}")
                return {'bleu_score': bleu_score}
            except Exception as e:
                print(f"Warning: Evaluation error: {str(e)}")
        return {}

def train_and_save_model():
    print("Starting training process...")
    start_time = time.time()
    
    # Initialize model and tokenizer
    model_name = "gpt2"
    print(f"Loading {model_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Add padding token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    # Load dataset
    print("Loading dataset...")
    raw_dataset = load_dataset("tiny_shakespeare", trust_remote_code=True)
    
    # Preprocess text
    print("Preprocessing text...")
    text = raw_dataset['train'][0]['text']
    cleaned_text = preprocess_text(text)
    
    # Create chunks with overlap
    chunk_size = 128
    overlap_size = 32
    chunks = create_training_chunks(cleaned_text, chunk_size, overlap_size)
    
    # Use 50% of data
    chunks = chunks[:int(len(chunks) * 0.5)]
    
    # Create training and evaluation splits (90-10)
    train_size = int(len(chunks) * 0.9)
    train_chunks = chunks[:train_size]
    eval_chunks = chunks[train_size:]
    
    print(f"Created {len(train_chunks)} training chunks and {len(eval_chunks)} evaluation chunks")
    
    # Create datasets
    train_dataset = Dataset.from_dict({'text': train_chunks})
    eval_dataset = Dataset.from_dict({'text': eval_chunks})
    
    # Calculate initial BLEU score
    print("Calculating initial BLEU score...")
    initial_bleu = evaluate_model_bleu(model, tokenizer, eval_chunks[:50])
    print(f"Initial BLEU score: {initial_bleu:.4f}")
    
    # Calculate dataset statistics
    print("Calculating dataset statistics...")
    total_tokens = 0
    total_characters = 0
    vocab_usage = set()
    
    for chunk in train_chunks:
        tokens = tokenizer.encode(chunk)
        total_tokens += len(tokens)
        total_characters += len(chunk)
        vocab_usage.update(tokens)
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=chunk_size,
            return_tensors="np"
        )
    
    print("Tokenizing dataset...")
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=4
    )
    tokenized_eval = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        num_proc=4
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./training_output",
        num_train_epochs=3,              # Increased from 2
        per_device_train_batch_size=8,   # Increased from 4
        gradient_accumulation_steps=4,    # Reduced from 8
        save_steps=200,
        save_total_limit=2,
        logging_steps=25,
        learning_rate=5e-5,              # Reduced for better stability
        warmup_ratio=0.05,               # Adjusted
        weight_decay=0.01,
        adam_epsilon=1e-8,
        max_grad_norm=0.5,               # Reduced for stability
        dataloader_num_workers=4,
        evaluation_strategy="steps",
        eval_steps=200
    )
    
    # Initialize trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        eval_texts=eval_chunks[:50],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        ),
    )
    
    # Train the model
    print("Training model...")
    try:
        training_results = trainer.train()
        training_success = True
        
        # Calculate final BLEU score
        final_bleu = evaluate_model_bleu(model, tokenizer, eval_chunks[:50])
        print(f"Final BLEU score: {final_bleu:.4f}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        training_success = False
        training_results = None
        final_bleu = None
    
    # Save the model and tokenizer
    print("Saving model and tokenizer...")
    model.save_pretrained("./trained_model")
    tokenizer.save_pretrained("./trained_model")
    
    # Calculate and save metrics
    training_time = time.time() - start_time
    
    metrics = {
        'training_metrics': {
            'final_loss': float(training_results.training_loss) if training_success else None,
            'initial_loss': trainer.training_loss_history[0] if trainer.training_loss_history else None,
            'final_batch_loss': trainer.training_loss_history[-1] if trainer.training_loss_history else None,
            'training_time': f"{training_time:.2f} seconds",
            'training_time_per_epoch': f"{training_time/2:.2f} seconds",
            'training_success': training_success,
            'initial_bleu': initial_bleu,
            'final_bleu': final_bleu,
            'bleu_history': trainer.bleu_history
        },
        'model_metrics': {
            'vocab_size': len(vocab_usage),
            'params_count': sum(p.numel() for p in model.parameters()),
        },
        'dataset_metrics': {
            'train_chunks': len(train_chunks),
            'eval_chunks': len(eval_chunks),
            'total_tokens': total_tokens,
            'total_characters': total_characters,
            'avg_tokens_per_chunk': total_tokens / len(train_chunks),
            'vocab_usage_percentage': len(vocab_usage) / tokenizer.vocab_size * 100,
            'chunk_size': chunk_size,
            'overlap_size': overlap_size
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open('./trained_model/training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\nTraining completed!")
    if training_success:
        print(f"Final loss: {training_results.training_loss:.4f}")
        print(f"Final BLEU score: {final_bleu:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    print("Model and metrics saved in ./trained_model/")

if __name__ == "__main__":
    train_and_save_model()