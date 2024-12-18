# Creative-Storytelling-Using-LLMs
Project to compare with various Pre-trained transformer models on different datasets


## Abstract
Creative storytelling using **Large Language Models (LLMs)** explores the capabilities of **GPT-2**, **BERTBase**, and **T5-Small** for generating diverse narratives. Fine-tuned on datasets like **Merve Poetry**, **Tiny Shakespeare**, **WikiText**, and **Guanaco**, the models were evaluated using **Perplexity**, **BLEU Score**, and **Training Loss**. Results show **GPT-2** excels in fluency, **BERTBase** in coherence, and **T5-Small** in task-specific adaptability. A **Flask-based frontend** was developed for real-time interaction. Future work aims to create a **custom transformer model** integrating strengths from all three models to enhance storytelling capabilities.

---

## Introduction
Storytelling is a fundamental form of communication, blending creativity and context to share ideas and narratives. Modern **Large Language Models (LLMs)**, such as **GPT-2**, **BERTBase**, and **T5-Small**, have revolutionized this domain by generating coherent, engaging, and contextually rich text.  
This project aims to compare these models by fine-tuning them on diverse datasets to evaluate their suitability for creative storytelling. A **Flask-based web application** was developed to demonstrate real-time narrative generation.

---

## Dataset Description
1. **Merve Poetry**  
   - **Size**: ~10,000 poetic texts.  
   - **Reason**: Evaluates models’ ability to generate rhythmic and structured poetic narratives.

2. **Tiny Shakespeare**  
   - **Size**: ~112 KB of Shakespearean text.  
   - **Reason**: Tests models' capacity to mimic stylistic and archaic language.

3. **WikiText**  
   - **Size**: ~100 million tokens.  
   - **Reason**: Measures performance in generating factual, structured, and well-organized text.

4. **Guanaco**  
   - **Size**: ~20,000 conversational text samples.  
   - **Reason**: Assesses models’ ability to create dialogue-driven and interactive stories.

---

## Experimental Setup
1. **Model Architectures**:
   - **GPT-2**: Autoregressive decoder for fluent and sequential text generation.
   - **BERTBase**: Bidirectional encoder pre-trained using Masked Language Modeling (MLM).
   - **T5-Small**: Encoder-decoder with task-specific prefixes for flexible text-to-text generation.

2. **Frameworks and Tools**:
   - **Hugging Face Transformers**: For loading and fine-tuning models.
   - **Google Colab**: Used for GPU/TPU resources during training.
   - **Flask**: For developing a real-time storytelling interface.

3. **Evaluation Metrics**:
   - **Perplexity**: Measures fluency and model confidence.
   - **BLEU Score**: Evaluates overlap between generated and reference text.
   - **Training Loss**: Monitors convergence during training.

---

## Conclusions
- **GPT-2**: Best overall performer, generating fluent and engaging narratives, especially for long-form storytelling.
- **BERTBase**: Excels in coherence and context comprehension, ideal for completing or refining existing text.
- **T5-Small**: Most efficient for task-specific generation, with low perplexity and high adaptability.
- **Overall**: GPT-2 is the recommended model, but a custom transformer combining all three models can enhance storytelling further.

---

## Future Work
Future work involves developing a **custom transformer architecture** combining the strengths of:
1. **GPT-2**: Autoregressive decoder for fluent text generation.
2. **BERTBase**: Bidirectional attention for enhanced context comprehension.
3. **T5-Small**: Task-specific prefixes for controlled and adaptable storytelling.

This hybrid model aims to improve storytelling quality, coherence, and flexibility for diverse applications.
