import numpy as np
import json
import torch
from collections import defaultdict

class BookEmbedding:
    _instance = None  # Singleton instance
    _embedding_cache = {}  # Cache for book embeddings
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(BookEmbedding, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self, embedding_dim=100):
        if not self.initialized:
            self.embedding_dim = embedding_dim
            self.book_embeddings = {}
            self.genre_embeddings = defaultdict(lambda: np.random.normal(0, 0.1, embedding_dim))
            self.initialized = True
    
    def load_genres(self, genres_file='data/genres.json'):
        """Load and process genre information"""
        if not self.genre_embeddings:  # Only load if not already loaded
            self.book_genres = {}
            with open(genres_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    self.book_genres[data['book_id']] = data['genres']
                    
                    # Update genre embeddings based on co-occurrence
                    genres = list(data['genres'].keys())
                    for genre in genres:
                        if genre not in self.genre_embeddings:
                            self.genre_embeddings[genre] = np.random.normal(0, 0.1, self.embedding_dim)
    
    def create_book_embedding(self, book_id):
        """Create embedding from book's genres"""
        # Check cache first
        if book_id in self._embedding_cache:
            return self._embedding_cache[book_id]
            
        # Convert book_id to string if not already
        str_book_id = str(book_id)
        
        # If we don't have genre data for this book, create a unique random embedding
        if str_book_id not in self.book_genres:
            # Use book_id hash as seed to ensure the same book always gets the same random embedding
            book_id_hash = hash(str_book_id) % 2147483647  # Max 32-bit integer
            np.random.seed(book_id_hash)
            embedding = np.random.normal(0, 0.1, self.embedding_dim)
            np.random.seed()  # Reset the seed
            
            # Normalize the embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding /= norm
            
            self._embedding_cache[book_id] = embedding
            return embedding
            
        # If we have genre data, create the embedding based on genres
        embedding = np.zeros(self.embedding_dim)
        genres = self.book_genres[str_book_id]
        total_weight = sum(genres.values())
        
        # Weighted sum of genre embeddings
        for genre, count in genres.items():
            weight = count / total_weight
            embedding += weight * self.genre_embeddings[genre]
        
        # Add a small amount of book-specific noise to make embeddings unique
        book_id_hash = hash(str_book_id) % 2147483647
        np.random.seed(book_id_hash)
        noise = np.random.normal(0, 0.01, self.embedding_dim)  # Small noise
        np.random.seed()  # Reset the seed
        embedding += noise
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        
        # Cache the result
        self._embedding_cache[book_id] = embedding
        return embedding
    
    def get_embedding(self, book_id):
        """Get or create book embedding"""
        if book_id not in self.book_embeddings:
            self.book_embeddings[book_id] = self.create_book_embedding(book_id)
        return self.book_embeddings[book_id]