import os
import numpy as np
import torch
import argparse
import random
import sys
import time
import pandas as pd
import glob
from datetime import datetime
import re
from collections import defaultdict
import json
from functools import lru_cache

from agents.dqn_agent import DQNAgent
from book_embedding import BookEmbedding

# Cache for user recommendations to prevent duplicate processing
user_recommendations_cache = {}
cache_timeout = 300  # Cache timeout in seconds

@lru_cache(maxsize=1000)
def get_user_history(user_id, data_dir="data", interactions_file="processed_interactions.csv", 
                    book_id_map_file="book_id_map.csv", user_id_map_file="user_id_map.csv"):
    """Get the reading history for a specific user"""
    
    interactions_path = os.path.join(data_dir, interactions_file)
    user_book_path = os.path.join(data_dir, "user_book.csv")  # Add path to user_book.csv
    book_id_map_path = os.path.join(data_dir, book_id_map_file)
    user_id_map_path = os.path.join(data_dir, user_id_map_file)
    
    # Load book ID mapping
    book_id_map = {}
    if os.path.exists(book_id_map_path):
        try:
            book_id_df = pd.read_csv(book_id_map_path)
            print(f"Loaded book ID mapping with {len(book_id_df)} entries")
            # Use correct column names from CSV
            book_id_map = dict(zip(book_id_df['book_id_csv'], book_id_df['book_id']))
        except Exception as e:
            print(f"Error loading book ID mapping: {e}")
    
    # Load user ID mapping
    user_id_map = {}
    reverse_user_id_map = {}  # For mapping from hash to numeric ID
    if os.path.exists(user_id_map_path):
        try:
            user_id_df = pd.read_csv(user_id_map_path)
            print(f"Loaded user ID mapping with {len(user_id_df)} entries")
            # Use correct column names from CSV
            user_id_map = dict(zip(user_id_df['user_id_csv'], user_id_df['user_id']))
            # Create reverse mapping (hash -> numeric ID)
            reverse_user_id_map = dict(zip(user_id_df['user_id'], user_id_df['user_id_csv']))
        except Exception as e:
            print(f"Error loading user ID mapping: {e}")
    
    # Get user prefix for direct file scanning - take first 8 characters
    user_prefix = user_id[:8]
    print(f"Will use user prefix for direct scanning: {user_prefix}")
    
    # Get real user ID for display
    real_user_id = user_id_map.get(user_id, user_id)
    print(f"Looking for user {real_user_id} (encoded: {user_id})")
    
    # Start with direct file scanning for our specific format - most reliable method
    user_history = []
    if os.path.exists(user_book_path):
        try:
            print(f"Directly scanning user_book.csv for user prefix {user_prefix}")
            with open(user_book_path, 'r') as f:
                # Skip header
                header = f.readline().strip()
                print(f"File header: {header}")
                
                line_num = 1
                matches_found = 0
                
                # Process each line
                for line in f:
                    line_num += 1
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if line starts with the user prefix
                    if line.startswith(user_prefix):
                        try:
                            parts = line.split()
                            if len(parts) >= 3:  # Should have user_id, book_id, rating
                                book_id = parts[1]
                                rating = float(parts[2])
                                
                                # Map from CSV id to real id if needed
                                real_book_id = book_id_map.get(book_id, book_id)
                                
                                user_history.append({
                                    'encoded_id': book_id,
                                    'real_id': real_book_id,
                                    'rating': rating
                                })
                                
                                matches_found += 1
                                if matches_found <= 5:  # Only print first 5 matches to avoid flooding logs
                                    print(f"Found matching line {line_num}: {line}")
                        except Exception as e:
                            print(f"Error parsing line '{line}' at line {line_num}: {e}")
            
            if user_history:
                print(f"Found {len(user_history)} books in user's history via direct file scanning")
                return user_history
        except Exception as e:
            print(f"Error directly scanning user_book.csv: {e}")
    
    # If direct scanning didn't work, try traditional methods
    # Get user's history from processed_interactions.csv
    try:
        if os.path.exists(interactions_path):
            print(f"Trying processed_interactions.csv for user {user_id}")
            # Read the first chunk to get column names
            first_chunk = pd.read_csv(interactions_path, nrows=5)
            user_id_col = 'user_id_encoded' if 'user_id_encoded' in first_chunk.columns else 'user_id'
            
            for chunk in pd.read_csv(interactions_path, chunksize=100000):
                user_data = chunk[chunk[user_id_col] == user_id]
                if len(user_data) > 0:
                    book_id_col = 'book_id_encoded' if 'book_id_encoded' in user_data.columns else 'book_id'
                    rating_col = 'rating_scaled' if 'rating_scaled' in user_data.columns else ('rating' if 'rating' in user_data.columns else None)
                    
                    for _, row in user_data.iterrows():
                        encoded_book_id = int(row[book_id_col])
                        real_book_id = book_id_map.get(encoded_book_id, encoded_book_id)
                        rating = float(row[rating_col]) if rating_col else None
                        
                        user_history.append({
                            'encoded_id': encoded_book_id,
                            'real_id': real_book_id,
                            'rating': rating
                        })
        else:
            print(f"Interactions file not found: {interactions_path}")
    except Exception as e:
        print(f"Error processing interactions: {e}")
    
    # Try pandas method as a last resort
    if not user_history and os.path.exists(user_book_path):
        try:
            print(f"Trying pandas to read user_book.csv for user {user_id}")
            # First try using pandas to parse the file normally
            for chunk in pd.read_csv(user_book_path, chunksize=100000):
                if 'user_id' in chunk.columns and 'book_id' in chunk.columns:
                    # Try with exact user_id match
                    if user_id in chunk['user_id'].values:
                        user_data = chunk[chunk['user_id'] == user_id]
                        print(f"Found user with exact ID {user_id} in user_book.csv via pandas")
                    else:
                        continue
                        
                    for _, row in user_data.iterrows():
                        book_id = row['book_id']
                        rating = float(row['rating']) if 'rating' in row else None
                        
                        # Map from CSV id to real id if needed
                        real_book_id = book_id_map.get(book_id, book_id)
                        
                        user_history.append({
                            'encoded_id': book_id,
                            'real_id': real_book_id,
                            'rating': rating
                        })
        except Exception as e:
            print(f"Error processing user_book.csv with pandas: {e}")
    
    if not user_history:
        print(f"No interactions found for user {real_user_id}")
        return None
    
    print(f"Found {len(user_history)} books in user's history")
    return user_history

def get_random_user(data_dir="data", interactions_file="processed_interactions.csv", sample_size=1000):
    """Get a random user ID from the interactions file"""
    
    interactions_path = os.path.join(data_dir, interactions_file)
    user_book_path = os.path.join(data_dir, "user_book.csv")  # Add path to user_book.csv
    
    user_sample = []
    
    # Try processed_interactions.csv first
    if os.path.exists(interactions_path):
        try:
            print(f"Looking for users in {interactions_path}")
            # Read the first chunk to get column names
            first_chunk = pd.read_csv(interactions_path, nrows=5)
            user_id_col = 'user_id_encoded' if 'user_id_encoded' in first_chunk.columns else 'user_id'
            
            # Sample user IDs
            for chunk in pd.read_csv(interactions_path, chunksize=100000):
                users_in_chunk = chunk[user_id_col].unique()
                user_sample.extend(users_in_chunk[:min(sample_size, len(users_in_chunk))])
                if len(user_sample) >= sample_size:
                    break
        except Exception as e:
            print(f"Error sampling users from {interactions_path}: {e}")
    else:
        print(f"Interactions file not found: {interactions_path}")
    
    # If no users found in processed_interactions.csv, try user_book.csv
    if len(user_sample) == 0 and os.path.exists(user_book_path):
        try:
            print(f"Looking for users in {user_book_path}")
            # Sample user IDs from user_book.csv
            for chunk in pd.read_csv(user_book_path, chunksize=100000):
                if 'user_id' in chunk.columns:
                    users_in_chunk = chunk['user_id'].unique()
                    user_sample.extend(users_in_chunk[:min(sample_size, len(users_in_chunk))])
                    if len(user_sample) >= sample_size:
                        break
        except Exception as e:
            print(f"Error sampling users from {user_book_path}: {e}")
    
    if not user_sample:
        print("No users found in any interactions file")
        return None
    
    # Select a random user
    random_user = random.choice(user_sample)
    print(f"Selected random user: {random_user}")
    return random_user

def create_state_from_history(user_history, state_dim=100, embedder=None):
    """Create initial state vector from user history using real embeddings"""
    if not user_history:
        # Return random state for empty history
        print("Warning: Creating state from empty history")
        return np.random.normal(0, 0.1, state_dim).astype(np.float32)
    
    # Initialize with zeros
    state = np.zeros(state_dim, dtype=np.float32)
    weights = []
    book_embeddings = []
    
    # Initialize book embeddings
    if embedder is None:
        embedder = BookEmbedding(embedding_dim=state_dim)
        embedder.load_genres()
    
    # Generate a unique hash for this user based on their complete reading history
    user_hash = ""
    for book in user_history:
        user_hash += str(book.get('encoded_id', '0')) + str(book.get('rating', '3.0'))
    user_seed = int(abs(hash(user_hash)) % 10000000)
    # Set a consistent seed for this user to ensure the same user always gets the same randomization
    np.random.seed(user_seed)
    
    # First pass: collect embeddings and compute variance
    for book in user_history:
        book_id = book.get('encoded_id')
        book_embedding = embedder.get_embedding(str(book_id))
        rating = book.get('rating', 3.0)
        
        # More weight to extreme ratings (1 or 5) to capture strong preferences
        if rating >= 4.5:  # User really liked this book
            weight = 2.5  # Increased for stronger preference signal
        elif rating >= 4.0:
            weight = 1.8  # Increased for clearer preference signal
        elif rating <= 1.5:  # User really disliked this book
            weight = 1.2  # Still consider for state, but as negative signal
            book_embedding = -0.9 * book_embedding  # Stronger negative signal
        else:
            weight = max(0.2, rating / 5.0)  # Increased minimum weight
            
        weights.append(weight)
        book_embeddings.append(book_embedding)
    
    # Compute weighted sum of book embeddings with additional personalization
    for i, (embedding, weight) in enumerate(zip(book_embeddings, weights)):
        # Recent books have more influence on recommendations - strengthened recency bias
        recency_factor = 1.0
        if len(user_history) > 3:
            # Apply stronger recency bias for longer histories
            position = i / len(user_history)  # Normalized position (0-1)
            recency_factor = 1.0 + 3.0 * position  # Increased from 2.0
            
        # Add user-specific noise to each book embedding
        book_noise = np.random.normal(0, 0.08, state_dim)  # Increased noise
        modified_embedding = embedding + book_noise
            
        state += modified_embedding * weight * recency_factor

    # Normalize state
    total_weight = sum(weights) if weights else 1.0
    state /= total_weight
    
    # Add user-specific noise with higher variance
    noise = np.random.normal(0, 0.15, state_dim)  # Increased from 0.1
    state += noise
    
    # Create user-specific patterns in the state vector
    # This creates "signature" patterns that are unique to each user
    sig_pattern_count = 5
    pattern_length = 10
    for i in range(sig_pattern_count):
        start_idx = (user_seed + i * 17) % (state_dim - pattern_length)
        # Create a unique pattern for this user
        pattern = np.sin(np.linspace(0, 3*np.pi, pattern_length) + (user_seed % 6.28))
        pattern *= 0.3  # Scale down to not overwhelm the real signal
        state[start_idx:start_idx+pattern_length] += pattern
    
    # Add some unique elements based on ratio of genres or ratings
    if len(user_history) > 0:
        # Use number of books as a feature - stronger signal
        history_size_factor = min(1.0, len(user_history) / 20.0)  # Normalized 0-1
        state[0] = 0.9 + history_size_factor * 0.9  # Increased for stronger effect
        
        # Average rating as a feature - stronger signal
        avg_rating = sum(book.get('rating', 3.0) for book in user_history) / len(user_history)
        normalized_rating = (avg_rating - 1.0) / 4.0  # Map 1-5 to 0-1
        state[1] = 0.9 + normalized_rating * 0.9  # Increased for stronger effect
        
    # Add user ID-based features to more dimensions
    if len(user_history) > 0:
        # Use all book IDs to create more user-specific signals
        combined_id = ''.join([str(book.get('encoded_id', '0'))[:4] for book in user_history[:5]])
        for i, char in enumerate(combined_id[:min(15, len(combined_id))]):
            if i + 2 < state_dim:
                state[i + 2] = (ord(char) % 10) / 5.0  # Stronger values (0-2 range)
    
    # Make some dimensions strongly positive/negative based on user_seed
    # This creates very different state vectors for different users
    for i in range(10, 30):  # Increased range
        if i < state_dim:
            if (user_seed + i) % 5 == 0:
                state[i] = 1.5  # Increased from 1.0
            elif (user_seed + i) % 5 == 1:
                state[i] = -1.5  # Increased from -1.0
            elif (user_seed + i) % 5 == 2:
                state[i] = 0.8  # New mid-positive value
            elif (user_seed + i) % 5 == 3:
                state[i] = -0.8  # New mid-negative value
    
    # Add user-specific transformations to larger sections of the state
    if state_dim >= 50:
        transform_type = user_seed % 4
        mid_section = state[30:50]
        if transform_type == 0:
            # Amplify
            state[30:50] = mid_section * 1.5
        elif transform_type == 1:
            # Invert
            state[30:50] = -mid_section
        elif transform_type == 2:
            # Shift
            state[30:50] = mid_section + 0.5
        elif transform_type == 3:
            # Rotate/shift pattern
            state[30:50] = np.roll(mid_section, user_seed % 10)
    
    print(f"Created state vector with shape {state.shape}, mean {state.mean():.4f}, std {state.std():.4f}")
    print(f"State vector uniqueness indicators - hash: {user_seed}, first few values: {state[:5]}")
    return state.astype(np.float32)

def update_state(state, selected_book_id, state_dim=100):
    """Update state based on selected book using real embeddings"""
    embedder = BookEmbedding(embedding_dim=state_dim)
    embedder.load_genres()
    
    # Get real book embedding
    book_embedding = embedder.get_embedding(str(selected_book_id))
    
    # Move state slightly toward book embedding
    update_strength = 0.1
    new_state = state + update_strength * (book_embedding - state)
    
    return new_state.astype(np.float32)

def get_model_info(model_path):
    """Extract model architecture information from checkpoint"""
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['qnetwork_local_state_dict']
    
    # Get dimensions from the state dict
    input_size = state_dict['model.0.weight'].shape[1]    # State dimension
    hidden_size_1 = state_dict['model.0.weight'].shape[0] # First hidden layer
    hidden_size_2 = state_dict['model.2.weight'].shape[0] # Second hidden layer
    action_size = state_dict['model.4.weight'].shape[0]   # Action space size
    
    return {
        'state_dim': input_size,
        'hidden_sizes': [hidden_size_1, hidden_size2],
        'action_size': action_size
    }

def create_genre_weights(user_history, genre_data):
    """
    Calculate genre weights based on user history.
    
    Args:
        user_history (list): User's reading history
        genre_data (dict): Dictionary mapping book IDs to genre information
        
    Returns:
        dict: Genre weights normalized to represent importance
    """
    # Count genre occurrences in history
    genre_counts = defaultdict(float)
    total_count = 0
    
    for book in user_history:
        book_id = str(book['encoded_id'])
        if book_id in genre_data:
            # Get rating as a weighting factor (default to 3 if not available)
            rating_weight = float(book.get('rating', 3)) / 3
            
            for genre, count in genre_data[book_id].items():
                # Weight genre by both count and user rating
                genre_counts[genre] += count * rating_weight
                total_count += count
    
    # If we have no genre data, return empty dict
    if total_count == 0:
        return {}
        
    # Normalize to get importance weights
    genre_weights = {genre: count/total_count for genre, count in genre_counts.items()}
    
    # Sort by weight for easier debugging
    sorted_weights = dict(sorted(genre_weights.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_weights

def get_genre_distribution(book_ids, genre_data):
    """
    Calculate genre distribution for a list of books.
    
    Args:
        book_ids (list): List of book IDs
        genre_data (dict): Dictionary mapping book IDs to genre information
        
    Returns:
        dict: Genre distribution as percentages
    """
    genre_counts = defaultdict(float)
    total_count = 0
    
    for book_id in book_ids:
        str_book_id = str(book_id)
        if str_book_id in genre_data:
            for genre, count in genre_data[str_book_id].items():
                genre_counts[genre] += count
                total_count += count
    
    # If we have no genre data, return empty dict
    if total_count == 0:
        return {}
        
    # Calculate percentages
    genre_distribution = {genre: (count / total_count) * 100 
                         for genre, count in genre_counts.items()}
    
    return genre_distribution

def calculate_genre_similarity(user_genres, rec_genres):
    """
    Calculate how well recommendation genres match user preferences.
    
    Args:
        user_genres (dict): User's genre preferences (percentages)
        rec_genres (dict): Genres in recommendations (percentages)
        
    Returns:
        float: Similarity score (0-1)
    """
    if not user_genres:
        return 0
    
    similarity = 0
    total_weight = 0
    
    for genre, user_pct in user_genres.items():
        rec_pct = rec_genres.get(genre, 0)
        # Higher score when percentages are closer
        match = 1 - min(abs(user_pct - rec_pct) / max(user_pct, 0.01), 1)
        # Weight by importance to user
        genre_weight = user_pct / 100
        similarity += match * genre_weight
        total_weight += genre_weight
    
    # Normalize
    if total_weight > 0:
        return similarity / total_weight
    else:
        return 0

def diversify_recommendations(recommendations, genre_data, genre_weights, max_genre_percentage=0.25):
    """
    Ensure no genre exceeds max_percentage of recommendations unless it's the top user genre.
    
    Args:
        recommendations (list): Initial list of recommendations
        genre_data (dict): Dictionary mapping book IDs to genre information
        genre_weights (dict): User's genre preferences as weights
        max_genre_percentage (float): Maximum percentage for any non-top genre
        
    Returns:
        list: Diversified recommendations
    """
    if not recommendations or not genre_data or not genre_weights:
        return recommendations
    
    print(f"Diversifying recommendations with max_genre_percentage={max_genre_percentage}")
    
    # Make a copy of recommendations to avoid modifying the original list
    original_recommendations = recommendations.copy()
    total_slots = len(original_recommendations)
    
    # Get top user genres (up to 3)
    top_user_genres = []
    if genre_weights:
        sorted_genres = sorted(genre_weights.items(), key=lambda x: x[1], reverse=True)
        top_user_genres = [genre for genre, _ in sorted_genres[:3]]
        print(f"Top user genres (protected): {', '.join(top_user_genres)}")
    
    # Get current genre distribution
    current_genre_dist = {}
    current_book_genres = {}
    for book_id in original_recommendations:
        str_book_id = str(book_id)
        if str_book_id in genre_data:
            current_book_genres[book_id] = genre_data[str_book_id]
            for genre, count in genre_data[str_book_id].items():
                if genre in current_genre_dist:
                    current_genre_dist[genre] += count
                else:
                    current_genre_dist[genre] = count
    
    # Identify overrepresented genres
    total_genre_count = sum(current_genre_dist.values())
    overrepresented = []
    
    if total_genre_count > 0:
        for genre, count in current_genre_dist.items():
            percentage = (count / total_genre_count) * 100
            
            # Get the user's preference for this genre (if any)
            user_preference = genre_weights.get(genre, 0) * 100
            
            # Special handling for fantasy genre which is often overrepresented
            fantasy_factor = 1.5 if genre == "fantasy, paranormal" else 1.0
            
            # Consider a genre overrepresented if:
            # 1. It's not a top user genre, AND
            # 2. Either:
            #    a. It exceeds max_genre_percentage, OR
            #    b. It's more than double the user's preference for that genre
            if genre not in top_user_genres and (
                percentage > max_genre_percentage * 100 * fantasy_factor or 
                (user_preference > 0 and percentage > user_preference * 2)
            ):
                overrepresented.append((genre, percentage))
    
    # Special check for fantasy genre which tends to dominate
    fantasy_pct = current_genre_dist.get("fantasy, paranormal", 0) / total_genre_count * 100 if total_genre_count > 0 else 0
    user_fantasy_pct = genre_weights.get("fantasy, paranormal", 0) * 100
    
    # If fantasy is highly overrepresented compared to user preference, add it to overrepresented list
    if fantasy_pct > 30 and fantasy_pct > user_fantasy_pct * 3 and ("fantasy, paranormal", fantasy_pct) not in overrepresented:
        print(f"Special handling: fantasy genre is overrepresented ({fantasy_pct:.1f}% vs {user_fantasy_pct:.1f}% in user history)")
        overrepresented.append(("fantasy, paranormal", fantasy_pct))
    
    if overrepresented:
        print(f"Overrepresented genres: {', '.join([f'{g} ({p:.1f}%)' for g, p in overrepresented])}")
    else:
        print("No overrepresented genres found.")
        return original_recommendations
    
    # Identify books that contribute most to overrepresented genres
    problematic_genres = [g for g, _ in overrepresented]
    book_scores = {}
    
    for book_id in original_recommendations:
        book_genres = current_book_genres.get(book_id, {})
        
        # Skip if book doesn't have genre data
        if not book_genres:
            book_scores[book_id] = 0
            continue
        
        # Calculate score based on contribution to overrepresented genres
        # Higher scores = more problematic (bigger contributor to overrepresentation)
        score = 0
        for genre in problematic_genres:
            if genre in book_genres:
                # Books with higher counts in problematic genres get higher scores
                score += book_genres[genre]
                
                # Further increase score if genre is highly overrepresented
                for g, p in overrepresented:
                    if g == genre and p > max_genre_percentage * 150:  # 50% over threshold
                        score *= 1.5  # Increase priority for removal
        
        # Reduce score for books with top user genres
        for genre in top_user_genres:
            if genre in book_genres:
                score *= 0.5  # Halve the score for each top user genre present
                
        book_scores[book_id] = score
    
    # Sort books by score (highest first - most problematic)
    sorted_books = sorted(original_recommendations, key=lambda b: book_scores[b], reverse=True)
    
    # Calculate how many books we need to replace
    # Start with about 30% of the most problematic books
    replacement_count = min(int(total_slots * 0.3), 5)  # At most 5 books to avoid dramatic changes
    books_to_replace = sorted_books[:replacement_count]
    kept_books = [b for b in original_recommendations if b not in books_to_replace]
    
    print(f"Replacing {len(books_to_replace)} books to improve genre balance")
    
    # Calculate underrepresented genres
    user_preferred_genres = {}
    for genre, weight in genre_weights.items():
        current_percentage = 0
        if genre in current_genre_dist:
            current_percentage = (current_genre_dist[genre] / total_genre_count) * 100
        
        target_percentage = weight * 100
        
        # If genre is significantly underrepresented compared to user preference
        if current_percentage < target_percentage * 0.7:  # Less than 70% of target
            user_preferred_genres[genre] = target_percentage - current_percentage
    
    # Sort underrepresented genres by the gap
    sorted_underrepresented = sorted(user_preferred_genres.items(), key=lambda x: x[1], reverse=True)
    if sorted_underrepresented:
        print(f"Underrepresented genres: {', '.join([f'{g} (gap: {gap:.1f}%)' for g, gap in sorted_underrepresented[:3]])}")
    
    # Find replacement books with better genre balance
    # Generate a pool of eligible replacement books (those not in original recommendations)
    replacement_candidates = []
    
    # First try to find books with underrepresented genres
    if sorted_underrepresented:
        # Get list of top 3 genres to prioritize
        priority_genres = [g for g, _ in sorted_underrepresented[:3]]
        
        # Scan nearby books (within +/- 1000 of our current range)
        book_range = 1000
        min_id = max(0, min(original_recommendations) - book_range)
        max_id = max(original_recommendations) + book_range
        
        for book_id in range(min_id, max_id + 1):
            # Skip if book is already in recommendations
            if book_id in original_recommendations:
                continue
                
            str_book_id = str(book_id)
            if str_book_id in genre_data:
                book_genres = genre_data[str_book_id]
                
                # Skip books with overrepresented genres
                has_overrepresented = False
                for genre in problematic_genres:
                    if genre in book_genres and book_genres[genre] > 0.5:  # Significant presence
                        has_overrepresented = True
                        break
                
                if has_overrepresented:
                    continue
                
                # Calculate score for book based on underrepresented genres
                score = 0
                for genre in priority_genres:
                    if genre in book_genres:
                        # Higher count in priority genre = better score
                        score += book_genres[genre] * 2
                        
                        # Further boost score based on how underrepresented the genre is
                        for g, gap in sorted_underrepresented:
                            if g == genre:
                                score *= (1 + gap/100)  # More underrepresented = higher score
                
                if score > 0:
                    replacement_candidates.append((book_id, score))
                    
                # Limit the number of candidates to avoid long processing
                if len(replacement_candidates) >= 100:
                    break
    
    # If we don't have enough candidates, add some random ones
    if len(replacement_candidates) < replacement_count * 3:  # Aim for 3x options
        # Get some random books from nearby IDs
        for _ in range(50):
            random_id = random.randint(min(original_recommendations), max(original_recommendations))
            if random_id not in original_recommendations and random_id not in [r[0] for r in replacement_candidates]:
                replacement_candidates.append((random_id, 0.1))  # Low score for random additions
    
    # Sort candidates by score (higher is better)
    sorted_candidates = sorted(replacement_candidates, key=lambda x: x[1], reverse=True)
    
    # Replace problematic books with better candidates
    replacements = [c[0] for c in sorted_candidates[:replacement_count]]
    
    # Create the final diversified list
    result = kept_books + replacements
    
    # Recalculate genre distribution
    new_genre_dist = {}
    total_count = 0
    
    for book_id in result:
        str_book_id = str(book_id)
        if str_book_id in genre_data:
            for genre, count in genre_data[str_book_id].items():
                if genre in new_genre_dist:
                    new_genre_dist[genre] += count
                else:
                    new_genre_dist[genre] = count
                total_count += count
    
    # Show top genres in new distribution
    if total_count > 0:
        print("\nNew genre distribution after diversification:")
        sorted_new_dist = sorted([(g, (c/total_count)*100) for g, c in new_genre_dist.items()], 
                                key=lambda x: x[1], reverse=True)
        for genre, percentage in sorted_new_dist[:5]:
            print(f"  {genre}: {percentage:.1f}%")
    
    return result[:total_slots]  # Ensure we return exactly the right number of recommendations

def parse_compound_genres(genre_data):
    """
    Split compound genres into individual components for more precise matching.
    
    Args:
        genre_data (dict): Dictionary mapping book IDs to genre information
        
    Returns:
        dict: Expanded genre data with individual genres
    """
    expanded_genres = {}
    
    for book_id, genres in genre_data.items():
        expanded_genres[book_id] = {}
        for genre_name, weight in genres.items():
            # Split compound genre into individual genres
            if ',' in genre_name:
                individual_genres = [g.strip() for g in genre_name.split(',')]
                # Distribute the weight among individual genres
                individual_weight = weight / len(individual_genres)
                for individual in individual_genres:
                    if individual in expanded_genres[book_id]:
                        expanded_genres[book_id][individual] += individual_weight
                    else:
                        expanded_genres[book_id][individual] = individual_weight
            else:
                expanded_genres[book_id][genre_name] = weight
    
    return expanded_genres

def load_genre_data(data_dir="data", split_compound_genres=True):
    """
    Load genre data from file, with option to split compound genres.
    
    Args:
        data_dir (str): Directory containing genre data
        split_compound_genres (bool): Whether to split compound genres into components
        
    Returns:
        dict: Genre data dictionary
    """
    genre_data = {}
    genres_file = os.path.join(data_dir, "genres.json")
    
    if not os.path.exists(genres_file):
        print(f"Warning: Could not find genre data at {genres_file}")
        return genre_data
    
    try:
        with open(genres_file, 'r') as f:
            line_count = 0
            for line in f:
                try:
                    line_count += 1
                    book_genre = json.loads(line.strip())
                    book_id = str(book_genre['book_id'])
                    genre_data[book_id] = book_genre['genres']
                    
                    # Print progress occasionally
                    if line_count % 100000 == 0:
                        print(f"Loaded genre data for {line_count} books...")
                except Exception:
                    # Skip malformed lines
                    pass
        
        print(f"Loaded genre data for {len(genre_data)} books")
        
        # Split compound genres if requested
        if split_compound_genres:
            print("Splitting compound genres into individual components...")
            expanded_genres = parse_compound_genres(genre_data)
            print(f"Expanded genre data from {len(set([g for genres in genre_data.values() for g in genres]))} to {len(set([g for genres in expanded_genres.values() for g in genres]))} unique genres")
            return expanded_genres
        
    except Exception as e:
        print(f"Error loading genre data: {e}")
    
    return genre_data

def recommend_for_history(user_history, num_recommendations=20, model_path='models/dqn_model.pth',
                          state_dim=100, action_space_size=2000, list_models=False, exploration=0.2, apply_genre_matching=True, 
                          split_compound_genres=True):
    """Generate recommendations based on user history"""
    
    # Extract book IDs from user history
    book_ids = [int(item['encoded_id']) for item in user_history]
    
    # If list_models is True, print the available model files and return
    if list_models:
        model_files = glob.glob('models/*.pth')
        print(f"Available model files: {model_files}")
        return []
    
    # Load the model
    try:
        with torch.no_grad():
            agent = load_model(model_path)
            
        if agent is None:
            print("Failed to load model")
            return get_default_recommendations(num_recommendations)
            
        # Get action space size from model
        state_dim = 100  # Default value
        action_space_size = 2000  # Default value
        
        # Attempt to infer state and action space from model
        if hasattr(agent, 'qnetwork_local'):
            # Try to get state dim from first layer
            try:
                first_layer = list(agent.qnetwork_local.modules())[1]
                if hasattr(first_layer, 'in_features'):
                    state_dim = first_layer.in_features
                    print(f"Updated state dimension to {state_dim} based on model architecture")
            except (IndexError, AttributeError) as e:
                print(f"Could not determine state dimension from model: {e}")
            
            # Try to get action space from last layer
            output_layer = None
            for module in agent.qnetwork_local.modules():
                if isinstance(module, torch.nn.Linear):
                    output_layer = module
                    break
        
            if output_layer is not None and hasattr(output_layer, 'out_features'):
                action_space_size = output_layer.out_features
                print(f"Updated action space size to {action_space_size} based on model architecture")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        # Return some popular books as fallback
        return get_default_recommendations(num_recommendations)
        
    # Create book embedding utility
    embedder = BookEmbedding(embedding_dim=state_dim)
    embedder.load_genres()
    
    # Create state from user history
    state = create_state_from_history(user_history, state_dim, embedder)
    if state is None:
        print("Failed to create valid state from user history")
        return get_default_recommendations(num_recommendations)
    
    # Convert state to tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_tensor = torch.FloatTensor(state).to(device)
    
    # Get Q-values from the model with error handling
    try:
        with torch.no_grad():
            q_values = agent.qnetwork_local(state_tensor).cpu().numpy()
        print(f"Successfully generated Q-values of shape {q_values.shape}")
        
        # Ensure q_values is 1D for consistent processing
        if len(q_values.shape) > 1:
            q_values = q_values.flatten()
            print(f"Flattened Q-values to shape {q_values.shape}")
            
    except Exception as e:
        print(f"Error generating Q-values: {e}")
        return get_default_recommendations(num_recommendations)
    
    # Find the indices of the top Q-values
    # Exclude books that are already in the user's history
    history_set = set(book_ids)
    
    # Check if the q_values array is smaller than the expected action space
    if len(q_values) < action_space_size:
        print(f"Q-values shape {q_values.shape} is smaller than expected action space {action_space_size}")
        print("Using the available Q-values for recommendations")
        # Only consider available indices
        valid_indices = [(i, q) for i, q in enumerate(q_values) if i not in history_set]
    else:
        valid_indices = [(i, q) for i, q in enumerate(q_values) if i not in history_set]
    
    if not valid_indices:
        print("No valid book indices found after filtering user history. Using default recommendations.")
        return get_default_recommendations(num_recommendations)
    
    # Sort by Q-value (descending order)
    valid_indices.sort(key=lambda x: x[1], reverse=True)
    
    # Select top recommendations
    recommendations = [idx for idx, _ in valid_indices[:num_recommendations]]
    
    # Load genre data if we're doing genre matching
    genre_data = {}
    if apply_genre_matching:
        genre_data = load_genre_data(split_compound_genres=split_compound_genres)
        
        if user_history and genre_data:
            # Calculate genre weights from user history
            genre_weights = create_genre_weights(user_history, genre_data)
            
            # Apply genre-based diversification
            recommendations = diversify_recommendations(
                recommendations=recommendations,
                genre_data=genre_data,
                genre_weights=genre_weights,
                max_genre_percentage=0.25
            )
    
    return recommendations[:num_recommendations]

def get_recommendations_for_logged_user(user_id, num_recommendations=20, model_path=None):
    """Get recommendations for a logged-in user with improved caching"""
    cache_key = f"{user_id}_{num_recommendations}"
    current_time = time.time()
    
    # Check cache first
    if cache_key in user_recommendations_cache:
        cached_data, timestamp = user_recommendations_cache[cache_key]
        if current_time - timestamp < cache_timeout:
            return cached_data
    
    try:
        # Get user history with caching
        history = get_user_history(user_id)
        
        if not history:
            print(f"No history found for user {user_id}, using default recommendations")
            return get_default_recommendations(num_recommendations)
            
        # Load model if not provided
        if model_path is None:
            model_path = get_latest_model_path()
        
        # Generate recommendations    
        recommendations = recommend_for_history(
            user_history=history,
            num_recommendations=num_recommendations,
            model_path=model_path,
            apply_genre_matching=True
        )
        
        # Cache the results
        user_recommendations_cache[cache_key] = (recommendations, current_time)
        
        return recommendations
        
    except Exception as e:
        print(f"Error generating recommendations for user {user_id}: {str(e)}")
        return get_default_recommendations(num_recommendations)

def get_default_recommendations(num=20):
    """Get default recommendations (popular books) when personalized recommendations fail"""
    print("Using default recommendations.")
    
    # Instead of a static list, we'll try to get popular books from the CSV file
    try:
        import pandas as pd
        import random
        
        # Try to get popular books from the database
        if os.path.exists('data/preprocessed_books.csv'):
            books_df = pd.read_csv('data/preprocessed_books.csv')
            print(f"Loaded preprocessed_books.csv with columns: {books_df.columns.tolist()}")
            
            if len(books_df) > 0:
                # Determine the book ID column
                id_column = None
                for possible_id in ['id', 'book_id', 'bookID']:
                    if possible_id in books_df.columns:
                        id_column = possible_id
                        print(f"Using '{id_column}' as the book ID column")
                        break
                
                if id_column is None:
                    print("Could not determine book ID column in CSV")
                    raise ValueError("No ID column found in CSV")
                
                # Filter books with high ratings (4+ stars)
                if 'average_rating' in books_df.columns:
                    high_rated = books_df[books_df['average_rating'] >= 4.0]
                    if len(high_rated) >= num * 2:  # Ensure we have enough to sample from
                        # Sample a selection of highly-rated books
                        sampled_books = high_rated.sample(n=min(num * 2, len(high_rated)))
                        # Sort by rating and take top books
                        sorted_books = sampled_books.sort_values('average_rating', ascending=False)
                        # Extract IDs
                        book_ids = sorted_books[id_column].astype(str).tolist()[:num]
                        print(f"Found {len(book_ids)} popular books from ratings data")
                        return [int(book_id) for book_id in book_ids if book_id.isdigit()]
                
                # If we can't filter by rating, just sample some books
                sampled_books = books_df.sample(n=min(num, len(books_df)))
                book_ids = sampled_books[id_column].astype(str).tolist()
                book_ids = [int(book_id) for book_id in book_ids if book_id.isdigit()]
                print(f"Found {len(book_ids)} books by random sampling")
                return book_ids
            else:
                print("preprocessed_books.csv is empty")
    except Exception as e:
        print(f"Error getting popular books from CSV: {e}")
    
    print("Using hardcoded popular book recommendations")
    # Fallback to a mixed list of different genre bestsellers
    # This adds more diversity instead of always recommending the same books
    popular_literature = [2767052, 5107, 5470, 5907, 24583, 320, 11870085, 18586]
    popular_fantasy = [23692271, 4214, 960, 18619684, 1885, 10964, 2429135]
    popular_scifi = [10818853, 10572, 12967, 19063, 11084145]
    popular_mystery = [23437156, 16299, 3126496, 4214]
    popular_romance = [15818661, 3234472, 1420064, 6307287]
    
    # Create a diverse list by taking from each genre
    diverse_list = []
    all_genres = [popular_literature, popular_fantasy, popular_scifi, popular_mystery, popular_romance]
    
    # Determine how many books to take from each genre
    books_per_genre = max(2, num // len(all_genres))
    
    # Add books from each genre
    for genre in all_genres:
        # Shuffle the genre list to get different books each time
        random.shuffle(genre)
        # Take up to books_per_genre books from this genre
        diverse_list.extend(genre[:books_per_genre])
    
    # If we need more books, add from any genre
    all_books = popular_literature + popular_fantasy + popular_scifi + popular_mystery + popular_romance
    random.shuffle(all_books)
    diverse_list.extend(all_books[:num - len(diverse_list)])
    
    # Ensure we have the right number of unique books
    unique_list = list(dict.fromkeys(diverse_list))
    if len(unique_list) < num:
        # Add some from all_books that aren't already in unique_list
        remaining = [b for b in all_books if b not in unique_list]
        unique_list.extend(remaining[:num - len(unique_list)])
    
    return unique_list[:num]

def get_latest_model_path(model_dir='models', pattern='dqn_*.pth'):
    """Get the path to the latest model file in the specified directory
    
    Args:
        model_dir (str): Directory to search for model files
        pattern (str): File pattern to match for model files
        
    Returns:
        str: Path to the latest model file, or default path if none found
    """
    import os
    import glob
    import time
    
    start_time = time.time()
    
    # Direct check for default model - most common case
    default_path = 'models/dqn_model.pth'
    
    # Track paths checked for debugging
    checked_paths = []
    
    # Check if the default path exists and is accessible
    if os.path.exists(default_path):
        if os.path.isfile(default_path):
            if os.access(default_path, os.R_OK):
                print(f"✓ Found default model at {default_path}")
                print(f"Model path resolution took {time.time() - start_time:.3f}s")
                return default_path
            else:
                print(f"⚠️ Default model at {default_path} exists but is not readable")
                checked_paths.append(f"{default_path} (not readable)")
        else:
            print(f"⚠️ {default_path} exists but is not a file")
            checked_paths.append(f"{default_path} (not a file)")
    else:
        print(f"⚠️ Default model at {default_path} not found")
        checked_paths.append(f"{default_path} (not found)")
    
    # Try to create the model directory if it doesn't exist
    if not os.path.exists(model_dir):
        try:
            os.makedirs(model_dir, exist_ok=True)
            print(f"📁 Created missing model directory: {model_dir}")
        except Exception as e:
            print(f"❌ Failed to create model directory '{model_dir}': {str(e)}")
    
    # Check if model directory exists and is accessible
    if not os.path.exists(model_dir):
        print(f"⚠️ Model directory '{model_dir}' does not exist")
        checked_paths.append(f"{model_dir}/ (directory not found)")
    elif not os.path.isdir(model_dir):
        print(f"⚠️ '{model_dir}' exists but is not a directory")
        checked_paths.append(f"{model_dir}/ (not a directory)")
    elif not os.access(model_dir, os.R_OK):
        print(f"⚠️ '{model_dir}' exists but is not readable")
        checked_paths.append(f"{model_dir}/ (directory not readable)")
    else:
        try:
            # Look for files matching pattern
            pattern_path = os.path.join(model_dir, pattern)
            print(f"🔍 Searching for models matching: {pattern_path}")
            model_files = glob.glob(pattern_path)
            checked_paths.append(f"{pattern_path} ({len(model_files)} matches)")
            
            # If not found, try any .pth file
            if not model_files:
                pth_pattern = os.path.join(model_dir, '*.pth')
                print(f"🔍 No models matching '{pattern}', trying any .pth file: {pth_pattern}")
                model_files = glob.glob(pth_pattern)
                checked_paths.append(f"{pth_pattern} ({len(model_files)} matches)")
                
            # If found, return most recent by modification time
            if model_files:
                # Filter for files that actually exist and are readable
                valid_files = [f for f in model_files if os.path.exists(f) and os.path.isfile(f) and os.access(f, os.R_OK)]
                
                if valid_files:
                    valid_files.sort(key=os.path.getmtime, reverse=True)
                    latest_model = valid_files[0]
                    print(f"✓ Found latest model: {latest_model}")
                    print(f"Model path resolution took {time.time() - start_time:.3f}s")
                    return latest_model
                else:
                    print(f"⚠️ Found {len(model_files)} model paths in '{model_dir}' but none are valid, accessible files")
            else:
                print(f"⚠️ No .pth files found in '{model_dir}'")
        except Exception as e:
            print(f"❌ Error searching in '{model_dir}': {str(e)}")
            checked_paths.append(f"{model_dir}/ (error: {str(e)})")
    
    # Try to find any .pth file in current directory or subdirectories
    print("🔍 Searching for .pth files in current directory and subdirectories...")
    try:
        pth_files = []
        for root, _, files in os.walk('.'):
            for f in files:
                if f.endswith('.pth'):
                    file_path = os.path.join(root, f)
                    if os.path.exists(file_path) and os.path.isfile(file_path) and os.access(file_path, os.R_OK):
                        pth_files.append(file_path)
        
        if pth_files:
            pth_files.sort(key=os.path.getmtime, reverse=True)
            latest_file = pth_files[0]
            print(f"✓ Found model: {latest_file}")
            print(f"Model path resolution took {time.time() - start_time:.3f}s")
            return latest_file
        else:
            print("⚠️ No .pth files found in any subdirectory")
    except Exception as e:
        print(f"❌ Error during subdirectory search: {str(e)}")
    
    # Log paths that were checked to help with debugging
    print("\n📋 Summary of paths checked:")
    for path in checked_paths:
        print(f"  - {path}")
    
    # Copy default model from a backup location if we have one
    backup_default = './backup/dqn_model.pth'
    if os.path.exists(backup_default) and os.path.isfile(backup_default) and os.access(backup_default, os.R_OK):
        try:
            import shutil
            os.makedirs(os.path.dirname(default_path), exist_ok=True)
            shutil.copy2(backup_default, default_path)
            print(f"✓ Restored default model from backup: {backup_default} -> {default_path}")
            print(f"Model path resolution took {time.time() - start_time:.3f}s")
            return default_path
        except Exception as e:
            print(f"❌ Error restoring from backup: {str(e)}")
    
    # No model found anywhere, return the default path
    print(f"⚠️ No model found. Using default path: {default_path}")
    print(f"Model path resolution took {time.time() - start_time:.3f}s")
    return default_path

def load_model(model_path):
    """Load the recommendation model from the specified path.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        object: The loaded model, or None if loading failed
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'qnetwork_local_state_dict' in checkpoint:
            # Extract dimensions from the weights
            local_state_dict = checkpoint['qnetwork_local_state_dict']
            input_layer_weight = local_state_dict.get('model.0.weight')
            hidden_layer1_weight = local_state_dict.get('model.2.weight')
            output_layer_weight = local_state_dict.get('model.4.weight')
            
            if input_layer_weight is not None and hidden_layer1_weight is not None and output_layer_weight is not None:
                state_dim = input_layer_weight.size(1)
                hidden_size1 = input_layer_weight.size(0)
                hidden_size2 = hidden_layer1_weight.size(0)
                action_size = output_layer_weight.size(0)
                
                print(f"Detected model architecture: [{state_dim} -> {hidden_size1} -> {hidden_size2} -> {action_size}]")
                
                # Create agent with matching architecture
                agent = DQNAgent(
                    state_size=state_dim,
                    action_size=action_size,
                    hidden_sizes=[hidden_size1, hidden_size2]
                )
                
                # Load weights
                agent.qnetwork_local.load_state_dict(local_state_dict)
                agent.qnetwork_local.eval()  # Set to evaluation mode
                print("Successfully loaded model")
                return agent
        
        print("Invalid model format")
        return None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Get book recommendations for a specific user based on history')
    parser.add_argument('--user', type=str, required=True, 
                       help='User ID (required)')
    parser.add_argument('--num', type=int, default=20, 
                       help='Number of recommendations to generate')
    parser.add_argument('--model', type=str, default='models/dqn_model.pth', 
                       help='Path to the model file')
    parser.add_argument('--data_dir', type=str, default='data', 
                       help='Data directory')
    parser.add_argument('--interactions', type=str, default='processed_interactions.csv', 
                       help='Interactions file')
    parser.add_argument('--book_map', type=str, default='book_id_map.csv', 
                       help='Book ID mapping file')
    parser.add_argument('--list_models', action='store_true', 
                       help='List available trained models')
    parser.add_argument('--model_dir', type=str, default='output', 
                       help='Directory to search for models')
    # Add new arguments with defaults
    parser.add_argument('--state_dim', type=int, default=100,
                       help='State dimension for the model')
    parser.add_argument('--action_space', type=int, default=2360650,
                       help='Action space size for the model')
    parser.add_argument('--no_genre_matching', action='store_true',
                       help='Disable genre-based matching and diversification')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*60)
    print("  📚  BOOK BUDDY - USER RECOMMENDATIONS  📚")
    print("="*60)
    
    # If just listing models, do that and exit
    if args.list_models:
        recommend_for_history([], list_models=True)
        print("\nTo generate recommendations, specify a user ID with --user")
        return
    
    # Get user history
    user_history = get_user_history(
        user_id=args.user,
        data_dir=args.data_dir,
        interactions_file=args.interactions,
        book_id_map_file=args.book_map
    )
    
    if not user_history:
        print(f"No history found for user {args.user}. Cannot generate recommendations.")
        return
    
    try:
        # Generate recommendations
        recommend_for_history(
            user_history=user_history,
            num_recommendations=args.num,
            model_path=args.model,
            list_models=args.list_models,
            apply_genre_matching=not args.no_genre_matching
        )
        
        print("\nThanks for using BookBuddy! Happy reading! 📚")
    except Exception as e:
        print(f"\nError generating recommendations: {e}")
        import traceback
        print("\nDetailed error information:")
        traceback.print_exc()
        print("\nTry using --list_models to see available models")
        print("Or specify a specific model with --model <path>")

if __name__ == "__main__":
    main()