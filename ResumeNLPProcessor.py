"""
Resume NLP Processing with spaCy, TF-IDF, Word2Vec, and BERT
=============================================================
This module handles the NLP processing of resumes, including:
1. Tokenization and Lemmatization using spaCy
2. Vectorization using TF-IDF, Word2Vec, and BERT
"""

import pandas as pd
import numpy as np
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from transformers import BertTokenizer, BertModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Union

# Load spaCy model - you can choose different sizes based on your needs
# 'en_core_web_sm' is smaller/faster, 'en_core_web_lg' has word vectors included
nlp = spacy.load('en_core_web_sm')

class ResumeNLPProcessor:
    """Class for processing resumes with NLP techniques"""
    
    def __init__(self):
        """Initialize the NLP processor"""
        self.nlp = nlp
        # Load BERT tokenizer and model
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        # Set model to evaluation mode
        self.bert_model.eval()
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters and extra whitespace
        
        Args:
            text: Raw text from resume
            
        Returns:
            Cleaned text
        """
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Convert to lowercase
        text = text.lower().strip()
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> Dict[str, Any]:
        """
        Process text with spaCy to get tokens and lemmas
        
        Args:
            text: Cleaned text from resume
            
        Returns:
            Dictionary with tokens and lemmas
        """
        # Process with spaCy
        doc = self.nlp(text)
        
        # Extract tokens (excluding stopwords and punctuation)
        tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        
        # Extract lemmas (normalized word forms)
        lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        
        # Get named entities (might be useful for extracting skills, education, etc.)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        return {
            'tokens': tokens,
            'lemmas': lemmas,
            'entities': entities,
            'processed_text': ' '.join(lemmas)
        }
    
    def vectorize_tfidf(self, documents: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Convert documents to TF-IDF vectors
        
        Args:
            documents: List of processed texts
            
        Returns:
            TF-IDF matrix and feature names
        """
        # Initialize the TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        
        # Fit and transform the documents
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
        
        # Get feature names
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        return tfidf_matrix, feature_names
    
    def train_word2vec(self, tokenized_documents: List[List[str]], vector_size: int = 100) -> Word2Vec:
        """
        Train a Word2Vec model on the tokenized documents
        
        Args:
            tokenized_documents: List of lists of tokens
            vector_size: Dimensionality of vectors
            
        Returns:
            Trained Word2Vec model
        """
        # Train Word2Vec model
        model = Word2Vec(
            sentences=tokenized_documents,
            vector_size=vector_size,
            window=5,
            min_count=1,
            workers=4
        )
        
        return model
    
    def get_doc_vectors_word2vec(self, tokenized_documents: List[List[str]], 
                                word2vec_model: Word2Vec) -> List[np.ndarray]:
        """
        Create document vectors by averaging Word2Vec word vectors
        
        Args:
            tokenized_documents: List of lists of tokens
            word2vec_model: Trained Word2Vec model
            
        Returns:
            List of document vectors
        """
        doc_vectors = []
        
        for doc in tokenized_documents:
            # Filter tokens that are in the model vocabulary
            valid_tokens = [word for word in doc if word in word2vec_model.wv.key_to_index]
            
            if len(valid_tokens) > 0:
                # Average the word vectors
                doc_vector = np.mean([word2vec_model.wv[word] for word in valid_tokens], axis=0)
            else:
                # If no valid tokens, use zero vector
                doc_vector = np.zeros(word2vec_model.vector_size)
                
            doc_vectors.append(doc_vector)
            
        return doc_vectors
    
    def get_bert_embeddings(self, text: str) -> np.ndarray:
        """
        Get BERT embeddings for a text
        
        Args:
            text: Text to encode
            
        Returns:
            BERT embedding vector
        """
        # Tokenize and prepare for BERT
        encoded_input = self.bert_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Get BERT embedding (without gradient calculation)
        with torch.no_grad():
            output = self.bert_model(**encoded_input)
        
        # Use the [CLS] token embedding as the document embedding
        # (This is a common approach for document-level embeddings)
        return output.last_hidden_state[:, 0, :].numpy().flatten()
    
    def get_doc_vectors_bert(self, documents: List[str]) -> List[np.ndarray]:
        """
        Create document vectors using BERT embeddings
        
        Args:
            documents: List of texts
            
        Returns:
            List of document vectors
        """
        return [self.get_bert_embeddings(doc) for doc in documents]
    
    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
    
    def process_resume_batch(self, resume_texts: List[str]) -> Dict[str, Any]:
        """
        Process a batch of resumes and return processed data
        
        Args:
            resume_texts: List of raw resume texts
            
        Returns:
            Dictionary with processed data
        """
        # Clean texts
        cleaned_texts = [self.clean_text(text) for text in resume_texts]
        
        # Tokenize and lemmatize
        processed_data = [self.tokenize_and_lemmatize(text) for text in cleaned_texts]
        
        # Extract processed texts and tokens
        processed_texts = [data['processed_text'] for data in processed_data]
        tokenized_texts = [data['tokens'] for data in processed_data]
        
        # Generate TF-IDF vectors
        tfidf_matrix, feature_names = self.vectorize_tfidf(processed_texts)
        
        # Train Word2Vec model
        w2v_model = self.train_word2vec(tokenized_texts)
        
        # Get document vectors using Word2Vec
        w2v_vectors = self.get_doc_vectors_word2vec(tokenized_texts, w2v_model)
        
        # Get BERT embeddings (this can be computationally expensive)
        # We'll process only a few documents for demonstration
        bert_vectors = self.get_doc_vectors_bert(processed_texts[:3])
        
        return {
            'processed_data': processed_data,
            'tfidf_matrix': tfidf_matrix,
            'tfidf_features': feature_names,
            'word2vec_model': w2v_model,
            'word2vec_vectors': w2v_vectors,
            'bert_vectors': bert_vectors
        }

    def visualize_vector_similarities(self, vectors: List[np.ndarray], labels: List[str] = None):
        """
        Visualize similarities between vectors
        
        Args:
            vectors: List of document vectors
            labels: Labels for the documents
        """
        n = len(vectors)
        similarity_matrix = np.zeros((n, n))
        
        # Compute pairwise similarities
        for i in range(n):
            for j in range(n):
                similarity_matrix[i, j] = self.compute_similarity(vectors[i], vectors[j])
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, annot=True, cmap='Blues',
                   xticklabels=labels if labels else [f"Doc {i}" for i in range(n)],
                   yticklabels=labels if labels else [f"Doc {i}" for i in range(n)])
        plt.title('Document Similarity Matrix')
        plt.tight_layout()
        plt.show()

    def compare_job_to_resumes(self, job_description: str, resume_texts: List[str], 
                              method: str = 'tfidf') -> List[Tuple[int, float]]:
        """
        Compare job description to resumes and return similarity scores
        
        Args:
            job_description: Job description text
            resume_texts: List of resume texts
            method: Vectorization method ('tfidf', 'word2vec', or 'bert')
            
        Returns:
            List of (resume_index, similarity_score) tuples sorted by similarity
        """
        # Clean and process job description
        clean_job = self.clean_text(job_description)
        job_processed = self.tokenize_and_lemmatize(clean_job)
        job_text = job_processed['processed_text']
        
        # Clean and process resumes
        clean_resumes = [self.clean_text(text) for text in resume_texts]
        resume_processed = [self.tokenize_and_lemmatize(text) for text in clean_resumes]
        resume_texts_processed = [data['processed_text'] for data in resume_processed]
        
        # Calculate similarity based on method
        if method == 'tfidf':
            # Combine job and resumes for TF-IDF
            all_texts = [job_text] + resume_texts_processed
            tfidf_matrix, _ = self.vectorize_tfidf(all_texts)
            
            # Extract job vector and resume vectors
            job_vector = tfidf_matrix[0]
            resume_vectors = tfidf_matrix[1:]
            
            # Calculate similarities
            similarities = [
                (i, cosine_similarity(job_vector, rv.reshape(1, -1))[0][0])
                for i, rv in enumerate(resume_vectors)
            ]
            
        elif method == 'word2vec':
            # Tokenize all documents
            job_tokens = job_processed['tokens']
            resume_tokens = [data['tokens'] for data in resume_processed]
            
            # Train Word2Vec on all documents
            all_tokens = [job_tokens] + resume_tokens
            w2v_model = self.train_word2vec(all_tokens)
            
            # Get document vectors
            all_vectors = self.get_doc_vectors_word2vec(all_tokens, w2v_model)
            job_vector = all_vectors[0]
            resume_vectors = all_vectors[1:]
            
            # Calculate similarities
            similarities = [
                (i, self.compute_similarity(job_vector, rv))
                for i, rv in enumerate(resume_vectors)
            ]
            
        elif method == 'bert':
            # Get BERT embeddings
            job_vector = self.get_bert_embeddings(job_text)
            resume_vectors = self.get_doc_vectors_bert(resume_texts_processed)
            
            # Calculate similarities
            similarities = [
                (i, self.compute_similarity(job_vector, rv))
                for i, rv in enumerate(resume_vectors)
            ]
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities


# Example usage
if __name__ == "__main__":
    # Sample data (in a real scenario, these would come from your database)
    sample_resumes = [
        """
        John Doe
        Software Engineer with 5 years of experience in Python and Machine Learning.
        Skills: Python, TensorFlow, PyTorch, SQL, Docker
        Experience: Senior Developer at Tech Co (2018-2023)
        Education: MS Computer Science, Stanford University
        """,
        
        """
        Jane Smith
        Data Scientist specializing in NLP and Deep Learning.
        Skills: Python, R, BERT, Word2Vec, spaCy, Keras
        Experience: Data Scientist at AI Labs (2019-2023)
        Education: PhD in Computational Linguistics, MIT
        """,
        
        """
        Bob Johnson
        Frontend Developer with UI/UX expertise.
        Skills: JavaScript, React, HTML, CSS, Figma, Adobe XD
        Experience: UI Developer at Design Agency (2017-2023)
        Education: BFA Digital Design, Rhode Island School of Design
        """
    ]
    
    sample_job = """
    We are looking for a Data Scientist with NLP expertise.
    Required skills: Python, Machine Learning, Deep Learning, NLP, BERT
    Responsibilities: Build text classification models, develop information extraction systems
    """
    
    # Initialize processor
    processor = ResumeNLPProcessor()
    
    # Process resumes
    results = processor.process_resume_batch(sample_resumes)
    
    # Compare job to resumes
    print("TF-IDF Similarity Ranking:")
    tfidf_rankings = processor.compare_job_to_resumes(sample_job, sample_resumes, method='tfidf')
    for idx, score in tfidf_rankings:
        print(f"Resume {idx}: {score:.4f}")
    
    print("\nWord2Vec Similarity Ranking:")
    w2v_rankings = processor.compare_job_to_resumes(sample_job, sample_resumes, method='word2vec')
    for idx, score in w2v_rankings:
        print(f"Resume {idx}: {score:.4f}")
    
    print("\nBERT Similarity Ranking:")
    bert_rankings = processor.compare_job_to_resumes(sample_job, sample_resumes, method='bert')
    for idx, score in bert_rankings:
        print(f"Resume {idx}: {score:.4f}")


        