import os
# Set tokenizers parallelism before importing libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SimpleQASystem:
    def __init__(self):
        """Initialize QA system using T5"""
        try:
            # Use T5 for answer generation
            self.model_name = 't5-small'
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, legacy=False)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

            # Move model to CPU explicitly to avoid memory issues
            self.device = "cpu"
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

            # Move the question embedding to CPU (if not already)
            question_embedding = question_embedding.cpu()

            # Find most similar answer using cosine similarity
            similarities = cosine_similarity(
                question_embedding.numpy().reshape(1, -1),  # Use .numpy() for numpy compatibility
                np.array([embedding.cpu().numpy() for embedding in self.answer_embeddings])  # Move answer embeddings to CPU
            )[0]

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


def main():
    """Main function with sample usage"""
    try:
        # Sample data
        data = [
            {"question": "What is the capital of France?", "answer": "The capital of France is Paris."},
            {"question": "What is the largest planet?", "answer": "The largest planet is Jupiter."},
            {"question": "Who wrote '1984'?", "answer": "George Orwell wrote '1984'."}
        ]

        # Initialize system
        print("Initializing QA system...")
        qa_system = SimpleQASystem()

        # Prepare dataset
        print("Preparing dataset...")
        qa_system.prepare_dataset(data)

        # Start interactive Q&A session
        while True:
            # Prompt the user for a question
            test_question = input("\nPlease enter your question (or 'exit' to quit): ")

            if test_question.lower() == 'exit':
                print("Exiting the program.")
                break

            # Get and print the answer
            print(f"\nQuestion: {test_question}")
            answer = qa_system.get_answer(test_question)
            print(f"Answer: {answer}")

    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()
