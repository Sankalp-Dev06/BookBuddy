import numpy as np
import pandas as pd
import torch
import random
import gym
from gym import spaces
import os
import scipy
from scipy import sparse
from typing import Tuple, Dict, Any, List, Optional


class RealBookRecommendationEnv(gym.Env):
    """
    A book recommendation environment that uses real book and user data.
    This environment follows the OpenAI Gym interface.
    """
    
    def __init__(self, 
                 state_dim: int = 100, 
                 action_space_size: int = 1000, 
                 max_steps: int = 10,
                 data_dir: str = 'data',
                 books_file: str = 'preprocessed_books.csv',
                 book_id_map_file: str = 'book_id_map.csv',
                 user_id_map_file: str = 'user_id_map.csv',
                 interactions_file: str = 'processed_interactions.csv',
                 genres_file: str = 'genres_sparse.npz',
                 user_sample_rate: float = 1.0,
                 max_users: int = 10000,
                 verbose: bool = True):
        """
        Initialize the real book recommendation environment.
        
        Args:
            state_dim: Dimension of the state vector
            action_space_size: Size of the action space (number of books to consider)
                              If None, will use all available books
            max_steps: Maximum steps per episode
            data_dir: Directory containing data files
            books_file: CSV file with book information
            book_id_map_file: CSV file with mapping between book IDs
            user_id_map_file: CSV file with mapping between user IDs
            interactions_file: CSV file with user-book interactions
            genres_file: NPZ file with sparse genre matrix
            user_sample_rate: Fraction of users to sample (0.0-1.0)
            max_users: Maximum number of users to keep in history
            verbose: Whether to print detailed logging information
        """
        super(RealBookRecommendationEnv, self).__init__()
        
        self.state_dim = state_dim
        self.action_space_size = action_space_size
        self.max_steps = max_steps
        self.data_dir = data_dir
        self.books_file = books_file
        self.book_id_map_file = book_id_map_file
        self.user_id_map_file = user_id_map_file
        self.interactions_file = interactions_file
        self.genres_file = genres_file
        self.user_sample_rate = user_sample_rate
        self.max_users = max_users
        self.verbose = verbose
        
        # Initialize environment state
        self.current_step = 0
        self.current_user_id = None
        self.current_user_encoded = None
        self.current_state = None
        self.recommended_books = set()
        
        # Load data
        self.books = None
        self.book_id_map = None
        self.user_id_map = None
        self.interactions = None
        self.genres = None
        self.book_embeddings = {}
        self.user_history = {}
        
        # Set up mappings
        self.book_id_to_encoded = {}
        self.encoded_to_book_id = {}
        self.user_id_to_encoded = {}
        self.encoded_to_user_id = {}
        
        # Set random seed
        self.np_random = None
        self.seed()
        
        # Load data
        self.load_data()
        
        # Determine actual action space size based on available books
        actual_action_space_size = len(self.book_embeddings)
        if self.action_space_size is None:
            self.action_space_size = actual_action_space_size
            if self.verbose:
                print(f"Using all available books: action_space_size = {self.action_space_size}")
        elif actual_action_space_size < self.action_space_size:
            self.action_space_size = actual_action_space_size
            if self.verbose:
                print(f"Limiting action space to available books: {self.action_space_size}")
        
        # Ensure action space is at least 1
        self.action_space_size = max(1, self.action_space_size)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.action_space_size)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        
        if self.verbose:
            print(f"Environment initialized with {self.action_space_size} books and state dimension {self.state_dim}")
    
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set random seeds for reproducibility"""
        self.np_random = np.random.RandomState(seed)
        return [seed]
    
    def load_data(self):
        """Load the data from files"""
        if self.verbose:
            print(f"Loading data from {self.data_dir}...")
        
        # File paths
        books_path = os.path.join(self.data_dir, self.books_file)
        book_id_map_path = os.path.join(self.data_dir, self.book_id_map_file)
        user_id_map_path = os.path.join(self.data_dir, self.user_id_map_file)
        interactions_path = os.path.join(self.data_dir, self.interactions_file)
        genres_path = os.path.join(self.data_dir, self.genres_file)
        
        # Load books data
        try:
            if os.path.exists(books_path):
                if self.verbose:
                    print(f"Loading books from {books_path}")
                self.books = pd.read_csv(books_path)
                if self.verbose:
                    print(f"Loaded {len(self.books)} books")
            else:
                if self.verbose:
                    print(f"Warning: Books file not found at {books_path}")
        except Exception as e:
            if self.verbose:
                print(f"Error loading books: {e}")
        
        # Load book ID mapping
        try:
            if os.path.exists(book_id_map_path):
                if self.verbose:
                    print(f"Loading book ID mapping from {book_id_map_path}")
                self.book_id_map = pd.read_csv(book_id_map_path)
                # Create mapping dictionaries
                if 'book_id' in self.book_id_map.columns and 'book_id_csv' in self.book_id_map.columns:
                    self.book_id_to_encoded = dict(zip(self.book_id_map['book_id'], self.book_id_map['book_id_csv']))
                    self.encoded_to_book_id = dict(zip(self.book_id_map['book_id_csv'], self.book_id_map['book_id']))
                    if self.verbose:
                        print(f"Created mapping for {len(self.book_id_to_encoded)} book IDs")
            else:
                if self.verbose:
                    print(f"Warning: Book ID mapping file not found at {book_id_map_path}")
        except Exception as e:
            if self.verbose:
                print(f"Error loading book ID mapping: {e}")
        
        # Load user ID mapping
        try:
            if os.path.exists(user_id_map_path):
                if self.verbose:
                    print(f"Loading user ID mapping from {user_id_map_path}")
                self.user_id_map = pd.read_csv(user_id_map_path)
                # Create mapping dictionaries
                if 'user_id' in self.user_id_map.columns and 'user_id_csv' in self.user_id_map.columns:
                    self.user_id_to_encoded = dict(zip(self.user_id_map['user_id'], self.user_id_map['user_id_csv']))
                    self.encoded_to_user_id = dict(zip(self.user_id_map['user_id_csv'], self.user_id_map['user_id']))
                    if self.verbose:
                        print(f"Created mapping for {len(self.user_id_to_encoded)} user IDs")
            else:
                if self.verbose:
                    print(f"Warning: User ID mapping file not found at {user_id_map_path}")
        except Exception as e:
            if self.verbose:
                print(f"Error loading user ID mapping: {e}")
        
        # Load interactions data
        try:
            if os.path.exists(interactions_path):
                if self.verbose:
                    print(f"Loading interactions from {interactions_path}")
                self.interactions = pd.read_csv(interactions_path)
                
                # Sample users if specified
                if self.user_sample_rate < 1.0:
                    unique_users = self.interactions['user_id_encoded'].unique()
                    num_users_to_sample = max(1, int(len(unique_users) * self.user_sample_rate))
                    sampled_users = self.np_random.choice(unique_users, num_users_to_sample, replace=False)
                    self.interactions = self.interactions[self.interactions['user_id_encoded'].isin(sampled_users)]
                    if self.verbose:
                        print(f"Sampled {num_users_to_sample} users from {len(unique_users)} total users")
                
                # Process interactions to create user history
                self._process_interactions()
                if self.verbose:
                    print(f"Processed {len(self.interactions)} interactions for {len(self.user_history)} users")
            else:
                if self.verbose:
                    print(f"Warning: Interactions file not found at {interactions_path}")
        except Exception as e:
            if self.verbose:
                print(f"Error loading interactions: {e}")
        
        # Load genres data
        try:
            if os.path.exists(genres_path) and genres_path.endswith('.npz'):
                if self.verbose:
                    print(f"Loading genres sparse matrix from {genres_path}")
                try:
                    # Try two different import methods to ensure it works
                    try:
                        self.genres = scipy.sparse.load_npz(genres_path)
                    except:
                        from scipy import sparse
                        self.genres = sparse.load_npz(genres_path)
                    
                    if self.verbose:
                        print(f"Loaded genres sparse matrix with shape {self.genres.shape}")
                except Exception as sparse_error:
                    if self.verbose:
                        print(f"Error loading sparse matrix: {sparse_error}")
                    print("Creating empty genres data")
                    self.genres = None
            else:
                if self.verbose:
                    print(f"Warning: Genres file not found or not an NPZ file at {genres_path}")
                self.genres = None
        except Exception as e:
            if self.verbose:
                print(f"Error in genre processing: {str(e)}")
            self.genres = None
        
        # Initialize book embeddings
        self._initialize_embeddings()
        
        if self.verbose:
            print(f"Data loading complete. Environment has {len(self.book_embeddings)} book embeddings.")
    
    def _process_interactions(self):
        """Process interactions from preprocessed data with user_id_encoded, book_id_encoded, rating_scaled format"""
        if self.interactions is None:
            if self.verbose:
                print("No interactions data to process")
            return
        
        if self.verbose:
            print("Processing interactions data...")
        
        # Get column names from the data
        if self.verbose:
            print(f"Available interaction columns: {list(self.interactions.columns)}")
        
        # Check for expected preprocessed format
        has_expected_format = (
            'user_id_encoded' in self.interactions.columns and 
            'book_id_encoded' in self.interactions.columns
        )
        
        if has_expected_format:
            if self.verbose:
                print("Found expected preprocessed format with user_id_encoded and book_id_encoded")
            user_col = 'user_id_encoded'
            book_col = 'book_id_encoded'
            
            # Check for rating column 
            if 'rating_scaled' in self.interactions.columns:
                rating_col = 'rating_scaled'
                if self.verbose:
                    print("Using rating_scaled for personalization")
            elif 'rating' in self.interactions.columns:
                rating_col = 'rating'
                if self.verbose:
                    print("Using rating column - scaling if needed")
                # Scale ratings to [0,1] if they're not already
                if self.interactions[rating_col].max() > 1:
                    if self.interactions[rating_col].max() <= 5:
                        if self.verbose:
                            print("Scaling ratings from 1-5 to 0-1 range")
                        self.interactions['rating_scaled'] = self.interactions[rating_col] / 5.0
                        rating_col = 'rating_scaled'
                    else:
                        if self.verbose:
                            print(f"Scaling ratings with max value {self.interactions[rating_col].max()}")
                        self.interactions['rating_scaled'] = self.interactions[rating_col] / self.interactions[rating_col].max()
                        rating_col = 'rating_scaled'
            else:
                if self.verbose:
                    print("No rating column found, using binary interactions")
                rating_col = None
        else:
            # Try to find suitable columns
            if self.verbose:
                print("Did not find expected preprocessed format, looking for usable columns")
            user_col = None
            for col in ['user_id', 'userId', 'user']:
                if col in self.interactions.columns:
                    user_col = col
                    if self.verbose:
                        print(f"Using {col} as user identifier")
                    break
            
            book_col = None
            for col in ['book_id', 'bookId', 'item_id', 'itemId', 'book', 'item']:
                if col in self.interactions.columns:
                    book_col = col
                    if self.verbose:
                        print(f"Using {col} as book identifier")
                    break
            
            rating_col = None
            for col in ['rating', 'rating_scaled', 'score', 'value']:
                if col in self.interactions.columns:
                    rating_col = col
                    if self.verbose:
                        print(f"Using {col} as rating")
                    break
        
        if user_col is None or book_col is None:
            if self.verbose:
                print("ERROR: Could not find user or book columns in interactions data")
                print(f"Available columns: {list(self.interactions.columns)}")
            return
        
        # Process interactions
        if self.verbose:
            print("Building user history from interactions...")
        n_users = 0
        n_interactions = 0
        
        # Import tqdm for progress bar if verbose
        if self.verbose:
            try:
                from tqdm import tqdm
                use_tqdm = True
            except ImportError:
                print("tqdm not installed, falling back to basic progress reporting")
                use_tqdm = False
        else:
            use_tqdm = False
        
        # Group users
        user_groups = self.interactions.groupby(user_col)
        total_users = len(user_groups)
        
        # Use tqdm progress bar if available and verbose
        if use_tqdm and self.verbose:
            iterator = tqdm(user_groups, desc="Processing users", total=total_users, unit="users")
        else:
            iterator = user_groups
            
        # Use more efficient processing by grouping by user
        for user_id, user_df in iterator:
            n_users += 1
            
            try:
                # Add each user's interactions to their history
                self.user_history[user_id] = []
                
                for _, row in user_df.iterrows():
                    try:
                        book_id = int(row[book_col])
                        
                        # Get rating if available, otherwise default to 1.0 (positive interaction)
                        if rating_col is not None:
                            rating = float(row[rating_col])
                            # Ensure rating is in [0,1]
                            rating = max(0.0, min(1.0, rating))
                        else:
                            rating = 1.0  # Default for binary interactions
                        
                        # Add to user history with book_id and rating
                        self.user_history[user_id].append((book_id, rating))
                        n_interactions += 1
                    except Exception as e:
                        if self.verbose and not use_tqdm:
                            print(f"Error processing interaction: {e}")
                
            except Exception as user_error:
                if self.verbose and not use_tqdm:
                    print(f"Error processing user {user_id}: {user_error}")
                continue
            
            # Print progress every 1000 users if not using tqdm
            if self.verbose and n_users % 1000 == 0 and not use_tqdm:
                print(f"Processed {n_users} users, {n_interactions} interactions")
        
        if self.verbose:
            print(f"Finished processing interactions: {n_users} users, {n_interactions} total interactions")
        
        # If the user history is too large, limit to most active users
        if self.max_users is not None and len(self.user_history) > self.max_users:
            if self.verbose:
                print(f"Large user base ({len(self.user_history)}). Limiting to {self.max_users} most active users.")
            users_by_activity = sorted(self.user_history.items(), key=lambda x: len(x[1]), reverse=True)
            self.user_history = dict(users_by_activity[:self.max_users])
            if self.verbose:
                print(f"Processed {n_interactions} interactions for {len(self.user_history)} users")
    
    def _initialize_embeddings(self):
        """Create book embeddings from available features"""
        # Clear existing embeddings
        self.book_embeddings = {}
        
        # If no books data, create random embeddings
        if self.books is None and self.action_space_size:
            if self.verbose:
                print("No book data found, creating random embeddings")
            for i in range(self.action_space_size):
                self.book_embeddings[i] = np.random.normal(0, 1, self.state_dim).astype(np.float32)
            return
        
        if self.books is not None:
            if self.verbose:
                print("Creating book embeddings from preprocessed features...")
            # First, check if we need to filter to only the books in our action space
            if self.action_space_size is not None and len(self.books) > self.action_space_size:
                if self.verbose:
                    print(f"Filtering to top {self.action_space_size} books")
                # Use most popular books or first n books
                valid_books = self.books.head(self.action_space_size)
            else:
                valid_books = self.books
                
            # Print column names for debugging
            if self.verbose:
                print(f"Available book columns: {list(valid_books.columns)}")
            
            # Process each book
            for idx, row in valid_books.iterrows():
                # Initialize embedding with zeros
                embedding = np.zeros(self.state_dim, dtype=np.float32)
                
                # Set a flag to track if we used any real features
                used_real_features = False
                book_id = None
                
                try:
                    # Get book ID (could be in different formats)
                    if 'book_id' in row:
                        book_id = int(row['book_id'])
                    elif 'id' in row:
                        book_id = int(row['id'])
                    else:
                        # If no ID column, use the index
                        book_id = int(idx)
                    
                    # Add book features (normalized) to the first few dimensions
                    feature_dim = 0
                    
                    # Average rating
                    if 'average_rating' in row and not pd.isna(row['average_rating']):
                        # Normalized to [0,1]
                        embedding[feature_dim] = float(row['average_rating'])
                        feature_dim += 1
                        used_real_features = True
                    
                    # Number of pages
                    if 'num_pages' in row and not pd.isna(row['num_pages']):
                        # Normalized to [0,1] using log scale
                        num_pages = float(row['num_pages'])
                        if num_pages > 0:
                            # Normalize pages: most books are between 100-1000 pages
                            embedding[feature_dim] = min(1.0, np.log10(num_pages) / 3.0)
                            feature_dim += 1
                            used_real_features = True
                    
                    # Author ID (if available)
                    if 'author_id' in row and not pd.isna(row['author_id']):
                        # Normalize author ID to a small value
                        author_id = int(row['author_id'])
                        # Use a hash of author ID for more uniform distribution
                        author_hash = hash(author_id) % 1000
                        embedding[feature_dim] = author_hash / 1000.0
                        feature_dim += 1
                        used_real_features = True
                    
                    # Add genre information if available
                    if self.genres is not None and book_id < self.genres.shape[0]:
                        try:
                            # Get the genre vector for this book
                            genre_vector = self.genres[book_id].toarray().flatten()
                            
                            # Add normalized genre vector to embedding
                            if len(genre_vector) > 0:
                                # Map genre vector to the available state dimensions
                                genre_end = min(feature_dim + len(genre_vector), self.state_dim)
                                embedding[feature_dim:genre_end] = genre_vector[:genre_end-feature_dim]
                                feature_dim = genre_end
                                used_real_features = True
                        except Exception as e:
                            if self.verbose:
                                print(f"Error adding genre features for book: {e}")
                    
                    # Fill the rest with random noise (unless we used all dimensions)
                    if feature_dim < self.state_dim:
                        noise = np.random.normal(0, 0.1, self.state_dim - feature_dim)
                        embedding[feature_dim:] = noise
                    
                    # Normalize the embedding
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    
                    # Only add if we have real features or are falling back to random
                    if used_real_features or len(self.book_embeddings) < self.action_space_size:
                        self.book_embeddings[book_id] = embedding
                except Exception as e:
                    if self.verbose:
                        print(f"Error adding book embedding: {e}")
        
        # Ensure we have at least one book embedding (fallback if no books loaded)
        if not self.book_embeddings:
            if self.verbose:
                print("WARNING: No book embeddings created. Creating random embeddings.")
            for i in range(min(1000, self.action_space_size or 1000)):
                self.book_embeddings[i] = np.random.normal(0, 1, self.state_dim).astype(np.float32)
        
        if self.verbose:
            print(f"Created {len(self.book_embeddings)} book embeddings")
    
    def get_user_history(self, user_encoded):
        """Get the user's reading history as a list of encoded book IDs"""
        if self.interactions is None or user_encoded not in self.interactions['user_id_encoded'].values:
            return []
        
        user_data = self.interactions[self.interactions['user_id_encoded'] == user_encoded]
        return user_data['book_id_encoded'].tolist()
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to start a new episode.
        
        Returns:
            numpy.ndarray: The initial state
        """
        # Reset episode variables
        self.current_step = 0
        self.recommended_books = set()
        
        # Select a random user
        if self.user_id_map is not None:
            self.current_user_encoded = self.np_random.choice(self.user_id_map['user_id_csv'])
            try:
                self.current_user_id = self.encoded_to_user_id[self.current_user_encoded]
            except KeyError:
                self.current_user_id = f"User_{self.current_user_encoded}"
        else:
            # Use a simulated user ID
            self.current_user_encoded = self.np_random.randint(1000)
            self.current_user_id = f"User_{self.current_user_encoded}"
        
        # Initialize state based on user profile
        self.current_state = self.generate_initial_state()
        
        return self.current_state
    
    def generate_initial_state(self) -> np.ndarray:
        """
        Generate the initial state representation for the current user.
        
        Returns:
            numpy.ndarray: Initial state vector
        """
        # Initialize state vector
        state = np.zeros(self.state_dim, dtype=np.float32)
        
        # Get user history
        user_history = self.get_user_history(self.current_user_encoded)
        
        if user_history:
            # Calculate average embedding of books the user has read
            embeddings = []
            for book_id in user_history[:10]:  # Limit to 10 most recent books for efficiency
                if book_id in self.book_embeddings:
                    embeddings.append(self.book_embeddings[book_id])
            
            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0)
                # Ensure the average embedding has the correct shape
                if avg_embedding.shape[0] < self.state_dim:
                    avg_embedding = np.pad(avg_embedding, (0, self.state_dim - avg_embedding.shape[0]))
                elif avg_embedding.shape[0] > self.state_dim:
                    avg_embedding = avg_embedding[:self.state_dim]
                
                state = avg_embedding
            else:
                # No valid embeddings found, use random state
                state = self.np_random.normal(0, 0.1, self.state_dim)
        else:
            # No user history, use random state
            state = self.np_random.normal(0, 0.1, self.state_dim)
        
        return state.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment by recommending a book.
        
        Args:
            action (int): The book index to recommend
            
        Returns:
            tuple:
                - numpy.ndarray: Next state
                - float: Reward
                - bool: Done flag
                - dict: Additional info
        """
        # Increment step counter
        self.current_step += 1
        
        # Validate action
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Track recommended books
        self.recommended_books.add(action)
        
        # Calculate reward based on user history and action
        reward = self.calculate_reward(action)
        
        # Update state based on the action and reward
        self.current_state = self.update_state(action, reward)
        
        # Check if episode is done
        done = (self.current_step >= self.max_steps)
        
        # Additional info
        info = {
            'step': self.current_step,
            'user_id': self.current_user_id,
            'user_encoded': self.current_user_encoded,
            'book_id': action,
            'recommended_books': list(self.recommended_books)
        }
        
        return self.current_state, reward, done, info
    
    def calculate_reward(self, action: int) -> float:
        """
        Calculate the reward for recommending a book.
        
        Args:
            action (int): The recommended book index
            
        Returns:
            float: Reward value
        """
        # If we've recommended this book before in this episode, apply very large penalty
        if action in self.recommended_books and len(self.recommended_books) > 1:
            return -2.0  # Increased penalty for duplicate recommendations
        
        # Check if the user has interacted with this book before
        user_history = self.get_user_history(self.current_user_encoded)
        
        if action in user_history:
            # Find the user's rating for this book
            user_rating = None
            if self.interactions is not None:
                user_book = self.interactions[(self.interactions['user_id_encoded'] == self.current_user_encoded) & 
                                             (self.interactions['book_id_encoded'] == action)]
                if not user_book.empty and 'rating_scaled' in user_book.columns:
                    user_rating = user_book.iloc[0]['rating_scaled']
            
            if user_rating is not None:
                # User has rated this book, return scaled rating [-1, 1]
                # Apply a steeper scaling to provide stronger signal
                # Rating of 0.5 (neutral) should give slightly negative reward
                scaled_rating = 2.5 * user_rating - 1.25  # This maps [0,1] → [-1.25,1.25]
                return max(min(scaled_rating, 1.0), -1.0)  # Clip to [-1,1] range
            else:
                # User has interacted with this book but no rating available
                # Slightly negative assumption since we don't know if they liked it
                return -0.1
        else:
            # For books the user hasn't interacted with, calculate similarity-based reward
            if action in self.book_embeddings:
                # Calculate similarity between user state and book embedding
                book_embedding = self.book_embeddings[action]
                sim_score = self.calculate_similarity(self.current_state, book_embedding)
                
                # Scale similarity to reward range with more conservative values
                # High similarity still needs exploration, so cap positive rewards
                reward = sim_score * 0.3  # Reduced scaling to prevent overconfidence
                
                # Add small noise to prevent getting stuck in local optima
                noise = self.np_random.normal(0, 0.05)
                reward += noise
                
                # Bound reward to slightly favor exploration (bias slightly negative)
                reward = max(min(reward, 0.8), -0.8)
                return reward
            else:
                # Unknown book, small negative reward to encourage exploration of known books
                return -0.2 + self.np_random.normal(0, 0.05)
    
    def calculate_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def update_state(self, action: int, reward: float) -> np.ndarray:
        """
        Update the state based on the action taken and reward received.
        
        Args:
            action (int): The book that was recommended
            reward (float): The reward received
            
        Returns:
            numpy.ndarray: Updated state
        """
        # Get book embedding
        if action in self.book_embeddings:
            book_embedding = self.book_embeddings[action]
        else:
            # Use a random embedding if the book is not in our database
            book_embedding = self.np_random.normal(0, 1, self.state_dim)
        
        # Update state: move slightly toward book embedding if reward is positive,
        # away if negative
        direction = 1 if reward > 0 else -1
        magnitude = abs(reward) * 0.1  # Small update factor
        
        # Update current state, moving toward or away from the book embedding
        updated_state = self.current_state + direction * magnitude * (book_embedding - self.current_state)
        
        # Add some exploration noise
        noise = self.np_random.normal(0, 0.01, self.state_dim)
        updated_state += noise
        
        return updated_state.astype(np.float32)
    
    def render(self, mode: str = 'human'):
        """
        Render the environment.
        
        Args:
            mode (str): Rendering mode
        """
        if mode == 'human':
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"User ID: {self.current_user_id} (Encoded: {self.current_user_encoded})")
            print(f"Recommended books: {self.recommended_books}")
            print(f"State summary: mean={np.mean(self.current_state):.4f}, std={np.std(self.current_state):.4f}")
        
    def close(self):
        """Clean up resources."""
        pass 

    def get_book_details(self, book_id):
        """Get detailed information about a book"""
        if self.books is None:
            return {"title": f"Book {book_id}", "authors": "Unknown"}
        
        book_info = {"title": f"Book {book_id}", "authors": "Unknown"}
        
        try:
            # Try to find by encoded ID first
            book_row = None
            
            # Check if this is an encoded ID that needs to be mapped
            real_book_id = None
            if hasattr(self, 'encoded_to_book_id') and book_id in self.encoded_to_book_id:
                real_book_id = self.encoded_to_book_id[book_id]
                
            # Try to find by encoded ID
            if 'book_id_encoded' in self.books.columns:
                book_row = self.books[self.books['book_id_encoded'] == book_id]
            
            # If not found and we have a real ID, try that
            if (book_row is None or book_row.empty) and real_book_id is not None:
                if 'book_id' in self.books.columns:
                    book_row = self.books[self.books['book_id'] == real_book_id]
            
            # If found, extract information
            if book_row is not None and not book_row.empty:
                if 'title' in book_row.columns:
                    book_info['title'] = book_row.iloc[0]['title']
                if 'authors' in book_row.columns:
                    book_info['authors'] = book_row.iloc[0]['authors']
                if 'average_rating' in book_row.columns:
                    book_info['average_rating'] = float(book_row.iloc[0]['average_rating'])
                if 'publication_year' in book_row.columns:
                    book_info['year'] = int(book_row.iloc[0]['publication_year'])
                if 'genres' in book_row.columns:
                    book_info['genres'] = book_row.iloc[0]['genres']
        except Exception as e:
            if self.verbose:
                print(f"Error retrieving book details for {book_id}: {e}")
        
        return book_info 