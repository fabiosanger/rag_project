import os
# Set tokenizers parallelism before importing libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .utils.text_processing import clean_answer
from .utils.data_loader import load_sample_data
from .utils.gpu_utils import get_gpu_device, is_gpu_available, move_to_device, clear_gpu_cache


class SimpleQASystem:
    def __init__(self):
        """Initialize QA system using T5"""
        try:
            # Use T5 for answer generation
            self.model_name = 't5-small'
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, legacy=False)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

            # Get optimal device (GPU if available, otherwise CPU)
            self.device = get_gpu_device()
            print(f"Using device: {self.device}")

            # Move model to the optimal device
            self.model = move_to_device(self.model, self.device)

            # Initialize storage
            self.answers = []
            self.answer_embeddings = None
            self.encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

            # Move encoder to the same device
            if is_gpu_available():
                self.encoder = move_to_device(self.encoder, self.device)

            print("System initialized successfully")

        except Exception as e:
            print(f"Initialization error: {e}")
            raise

    def prepare_dataset(self, data: List[Dict[str, str]]):
        """Prepare the dataset by storing answers and their embeddings"""
        try:
            # Store answers
            self.answers = [item['answer'] for item in data]

            # Encode answers using SentenceTransformer
            self.answer_embeddings = []
            for answer in self.answers:
                embedding = self.encoder.encode(answer, convert_to_tensor=True)
                # Move embedding to the same device as the model
                embedding = move_to_device(embedding, self.device)
                self.answer_embeddings.append(embedding)

            print(f"Prepared {len(self.answers)} answers")

        except Exception as e:
            print(f"Dataset preparation error: {e}")
            raise

    def get_answer(self, question: str) -> str:
        """Get answer using semantic search and T5 generation"""
        try:
            if not self.answers or self.answer_embeddings is None:
                raise ValueError("Dataset not prepared. Call prepare_dataset first.")

            # Encode question using SentenceTransformer
            question_embedding = self.encoder.encode(
                question,
                convert_to_tensor=True,
                show_progress_bar=False
            )

            # Move question embedding to the same device
            question_embedding = move_to_device(question_embedding, self.device)

            # Find most similar answer using cosine similarity
            # Convert tensors to CPU for sklearn compatibility
            question_cpu = question_embedding.cpu().numpy().reshape(1, -1)
            answers_cpu = np.array([embedding.cpu().numpy() for embedding in self.answer_embeddings])

            similarities = cosine_similarity(question_cpu, answers_cpu)[0]

            best_idx = np.argmax(similarities)
            context = self.answers[best_idx]

            # Generate the input text for the T5 model
            input_text = f"Given the context, what is the answer to the question: {question} Context: {context}"
            print(input_text)

            # Tokenize input text
            input_ids = self.tokenizer(
                input_text,
                max_length=512,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            ).input_ids.to(self.device)

            # Generate answer with limited max_length
            with torch.no_grad():  # Disable gradient computation for inference
                outputs = self.model.generate(
                    input_ids,
                    max_length=50,  # Increase length to handle more detailed answers
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )

            # Decode the generated answer
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Print the raw generated answer for debugging
            print(f"Generated answer before cleaning: {answer}")

            # Clean up the answer using utility function
            cleaned_answer = clean_answer(answer)

            # Clear GPU cache if using GPU
            if is_gpu_available():
                clear_gpu_cache()

            return cleaned_answer

        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"Error: {str(e)}"