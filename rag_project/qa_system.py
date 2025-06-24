import os
# Set tokenizers parallelism before importing libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SimpleQASystem:
    def __init__(self):
        """Initialize QA system using T5"""
        try:
            # Use T5 for answer generation - try different possible paths
            possible_paths = [
                'models/models-t5-small/snapshots/df1b051c49625cf57a3d0d8d3863ed4d13564fe4',
                '/home/fg12/repos/rag_project/models-t5-small/snapshots/df1b051c49625cf57a3d0d8d3863ed4d13564fe4',
                't5-small'  # Fallback to download from HuggingFace
            ]

            model_path = None
            for path in possible_paths:
                try:
                    if path == 't5-small':
                        # Try to load from HuggingFace Hub
                        self.tokenizer = T5Tokenizer.from_pretrained(path, legacy=False)
                        self.model = T5ForConditionalGeneration.from_pretrained(path)
                        model_path = path
                        break
                    else:
                        # Try to load from local path
                        self.tokenizer = T5Tokenizer.from_pretrained(path, legacy=False)
                        self.model = T5ForConditionalGeneration.from_pretrained(path)
                        model_path = path
                        break
                except Exception as e:
                    print(f"Failed to load model from {path}: {e}")
                    continue

            if model_path is None:
                raise ValueError("Could not load T5 model from any available path")

            # Extract a nice display name
            if 't5-small' in model_path:
                self.model_name = 'T5-Small'
            elif 't5-base' in model_path:
                self.model_name = 'T5-Base'
            elif 't5-large' in model_path:
                self.model_name = 'T5-Large'
            else:
                self.model_name = 'T5 Model'

            print(f"✅ Loaded model from: {model_path}")

            # Detect and set optimal device (GPU if available, otherwise CPU)
            if torch.cuda.is_available():
                try:
                    # Test CUDA functionality
                    test_tensor = torch.tensor([1.0], device='cuda')
                    del test_tensor
                    torch.cuda.empty_cache()

                    self.device = torch.device('cuda')
                    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                except Exception as e:
                    print(f"CUDA available but failed to initialize: {e}")
                    self.device = torch.device('cpu')
                    print("Falling back to CPU")
            else:
                self.device = torch.device('cpu')
                print("No GPU available, using CPU")

            # Move model to the detected device
            self.model = self.model.to(self.device)

            # Initialize storage
            self.answers = []
            self.answer_embeddings = None
            self.encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

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
                # Move to the same device as the model
                embedding = embedding.to(self.device)
                self.answer_embeddings.append(embedding)

            print(f"Prepared {len(self.answers)} answers")

        except Exception as e:
            print(f"Dataset preparation error: {e}")
            raise

    def clean_answer(self, answer: str) -> str:
        """Clean up generated answer by removing duplicates and extra whitespace"""
        words = answer.split()
        cleaned_words = []
        for i, word in enumerate(words):
            if i == 0 or word.lower() != words[i-1].lower():
                cleaned_words.append(word)
        cleaned = ' '.join(cleaned_words)
        return cleaned[0].upper() + cleaned[1:] if cleaned else cleaned

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

            # Move to the same device as answer embeddings
            question_embedding = question_embedding.to(self.device)

            # Stack answer embeddings into a single tensor
            answer_embeddings_tensor = torch.stack(self.answer_embeddings).to(self.device)

            # Calculate cosine similarity using PyTorch (GPU-compatible)
            similarities = F.cosine_similarity(
                question_embedding.unsqueeze(0),  # Add batch dimension
                answer_embeddings_tensor,
                dim=1
            )

            best_idx = similarities.argmax().item()
            context = self.answers[best_idx]

            # Generate the input text for the T5 model
            input_text = f"translate question to answer: {question} based on: {context}"
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
            with torch.no_grad():  # Add missing torch.no_grad() for inference
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

            # Clean up the answer
            cleaned_answer = self.clean_answer(answer)
            return cleaned_answer

        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"Error: {str(e)}"

    def get_answer_with_confidence(self, question: str, max_length: int = 50, num_beams: int = 4):
        """Get answer with confidence score using semantic search and T5 generation"""
        try:
            if not self.answers or self.answer_embeddings is None:
                raise ValueError("Dataset not prepared. Call prepare_dataset first.")

            # Encode question using SentenceTransformer
            question_embedding = self.encoder.encode(
                question,
                convert_to_tensor=True,
                show_progress_bar=False
            )

            # Move to the same device as answer embeddings
            question_embedding = question_embedding.to(self.device)

            # Stack answer embeddings into a single tensor
            answer_embeddings_tensor = torch.stack(self.answer_embeddings).to(self.device)

            # Calculate cosine similarity using PyTorch (GPU-compatible)
            similarities = F.cosine_similarity(
                question_embedding.unsqueeze(0),  # Add batch dimension
                answer_embeddings_tensor,
                dim=1
            )

            best_idx = similarities.argmax().item()

            # Calculate confidence score
            best_score = similarities[best_idx].item()
            confidence = max(0, min(100, best_score * 100))  # Convert to percentage

            context = self.answers[best_idx]

            # Generate the input text for the T5 model
            input_text = f"translate question to answer: {question} based on: {context}"
            print(input_text)

            # Tokenize input text
            input_ids = self.tokenizer(
                input_text,
                max_length=512,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            ).input_ids.to(self.device)

            # Generate answer with configurable parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode the generated answer
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Clean up the answer
            cleaned_answer = self.clean_answer(answer)

            return context, confidence, similarities.cpu().numpy()

        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"Error: {str(e)}", 0.0, []