import pandas as pd
import numpy as np
import torch
from agents.dqn_agent import DQNAgent
from book_embedding import BookEmbedding
import random

class RecommenderQNetwork(torch.nn.Module):
    """Q-Network matching the saved model architecture"""
    def __init__(self, state_size, action_size):
        super(RecommenderQNetwork, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_size, 64),    # First layer: 100 -> 64
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),            # Second layer: 64 -> 32
            torch.nn.ReLU(),
            torch.nn.Linear(32, action_size)     # Output layer: 32 -> action_size
        )
    
    def forward(self, state):
        return self.model(state)

def get_book_recommendations(book_id, num_recommendations=10):
    """Get recommendations for a given book using genre matching and DQN model"""
    try:
        # Convert book_id to string for consistency
        book_id_str = str(book_id)
        print(f"\nGenerating recommendations for book ID: {book_id_str}")
        
        genre_recommendations = []
        
        # Step 1: Genre-based recommendations (primary method)
        try:
            # Get book embedding and genre data
            embedding = BookEmbedding()
            embedding.load_genres()
            
            source_book_genres = embedding.book_genres.get(book_id_str, {})
            
            if source_book_genres and sum(source_book_genres.values()) > 0:
                print(f"Book has genre data with {len(source_book_genres)} genres")
                # Get genre-based recommendations
                genre_recommendations = get_genre_based_recommendations(
                    book_id_str, 
                    source_book_genres, 
                    embedding.book_genres, 
                    num_recommendations * 2  # Get more than needed to filter
                )
                
                # If we got enough genre recommendations, use them
                if len(genre_recommendations) >= num_recommendations:
                    print(f"Using genre-based recommendations for book {book_id_str}")
                    # Convert to the expected format of recommendation IDs and scores
                    final_recommendations = np.array([int(rec[0]) for rec in genre_recommendations])
                    final_scores = np.array([rec[1] for rec in genre_recommendations])
                    return final_recommendations, final_scores
                else:
                    print(f"Not enough genre recommendations ({len(genre_recommendations)}), trying model")
            else:
                print(f"No valid genre data for book {book_id_str} or zero weights, using model-based recommendations")
        except Exception as genre_error:
            print(f"Error getting genre-based recommendations: {genre_error}")
            import traceback
            traceback.print_exc()
            # Continue to model-based recommendations
            
        # Step 2: Fallback to model-based recommendations
        try:
            # Setup device and model path
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_path = 'models/dqn_model.pth'
            print(f"Using DQN model from: {model_path}")
            
            # Create network with correct architecture
            network = RecommenderQNetwork(state_size=100, action_size=2000)
            
            # Load the checkpoint and network state
            checkpoint = torch.load(model_path, map_location=device)
            network.load_state_dict(checkpoint['qnetwork_local_state_dict'])
            network = network.to(device)
            network.eval()
            
            # Get book embedding
            if 'embedding' not in locals():
                embedding = BookEmbedding()
                embedding.load_genres()
                
            # Create a significantly modified embedding for this specific book
            book_state = embedding.get_embedding(book_id)
            
            # Create a book-specific random generator
            book_id_hash = hash(book_id_str) % 2147483647
            rng = np.random.RandomState(book_id_hash)
            
            # Generate strong book-specific influence
            book_influence = rng.normal(0, 0.3, size=len(book_state))
            
            # Mix original embedding with strong book-specific influence
            modified_state = book_state * 0.6 + book_influence * 0.4
            
            # Re-normalize
            norm = np.linalg.norm(modified_state)
            if norm > 0:
                modified_state = modified_state / norm
                
            # Get model predictions
            state = torch.FloatTensor(modified_state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                q_values = network(state)
                # Get many recommendations to ensure diversity
                top_k_values, top_k_indices = torch.topk(q_values, 300)
                
                recommendations = top_k_indices[0].cpu().numpy()
                q_values = top_k_values[0].cpu().numpy()
                
            # Combine model and genre recommendations if we have both
            if len(genre_recommendations) > 0:
                print("Combining genre and model recommendations")
                # Get IDs from genre recommendations
                genre_rec_ids = [int(rec[0]) for rec in genre_recommendations]
                
                # Prioritize any recommendations that appear in both methods
                overlap_ids = [rec_id for rec_id in recommendations if rec_id in genre_rec_ids]
                
                # Then add remaining genre recommendations
                remaining_genre = [rec_id for rec_id in genre_rec_ids if rec_id not in overlap_ids]
                
                # Then add remaining model recommendations
                remaining_model = [rec_id for rec_id in recommendations if rec_id not in overlap_ids and rec_id not in genre_rec_ids]
                
                # Combine with priority: overlap first, then remaining from each
                final_ids = overlap_ids + remaining_genre + remaining_model
                
                # Create matching scores (higher for prioritized recommendations)
                scores = np.zeros(len(final_ids))
                scores[:len(overlap_ids)] = 0.9  # Highest score for overlap
                scores[len(overlap_ids):len(overlap_ids)+len(remaining_genre)] = 0.7  # High for genre
                scores[len(overlap_ids)+len(remaining_genre):] = 0.5  # Lower for model-only
                
                # Limit to requested number and convert to arrays
                final_recommendations = np.array(final_ids[:num_recommendations])
                final_scores = np.array(scores[:num_recommendations])
            else:
                # Use model recommendations only
                print("Using only model-based recommendations")
                # Add randomness based on book_id
                top_n = 150  # Consider more recommendations
                top_recommendations = recommendations[:top_n]
                top_q_values = q_values[:top_n]
                
                # Normalize q_values
                normalized_q = (top_q_values - np.min(top_q_values)) / (np.max(top_q_values) - np.min(top_q_values) + 1e-10)
                
                # Add significant noise based on book ID
                noise = rng.random(top_n) * 0.9  # Very high noise factor
                shuffled_scores = normalized_q * 0.4 + noise * 0.6  # More weight to randomness
                
                # Sort and reorder
                shuffled_indices = np.argsort(-shuffled_scores)
                reordered_recommendations = top_recommendations[shuffled_indices]
                reordered_q_values = top_q_values[shuffled_indices]
                
                # Use these as final recommendations
                final_recommendations = reordered_recommendations[:num_recommendations]
                final_scores = reordered_q_values[:num_recommendations]
            
            # Make sure the source book itself isn't in the recommendations
            final_recommendations = np.array([rec for rec in final_recommendations if str(rec) != book_id_str])
            
            # Ensure we have enough recommendations
            if len(final_recommendations) < num_recommendations and len(recommendations) > 0:
                # Add more from original recommendations if needed
                additional = [rec for rec in recommendations if rec not in final_recommendations and str(rec) != book_id_str]
                needed = num_recommendations - len(final_recommendations)
                final_recommendations = np.append(final_recommendations, additional[:needed])
            
            return final_recommendations, np.ones(len(final_recommendations))  # Placeholder scores
            
        except Exception as model_error:
            print(f"Error using model-based recommendations: {model_error}")
            import traceback
            traceback.print_exc()
            
        # Step 3: Absolute fallback - fixed popular books if both methods fail
        print("Both recommendation methods failed, using hardcoded popular books")
        # Return some bestsellers from different genres as an absolute fallback
        fallback_ids = [2767052, 5107, 5470, 960, 23692271, 4214, 10964, 10818853, 23437156, 15818661]
        random.shuffle(fallback_ids)  # Shuffle to avoid same recommendations every time
        
        # Create a unique shuffle based on book_id to ensure different books get different recommendations
        if not isinstance(book_id, str):
            book_id = str(book_id)
        seed = sum(ord(c) for c in book_id) % 10000
        random.seed(seed)
        random.shuffle(fallback_ids)
        random.seed()  # Reset the seed
        
        fallback_ids = fallback_ids[:num_recommendations]
        return np.array(fallback_ids), np.ones(len(fallback_ids))
        
    except Exception as e:
        print(f"Error getting recommendations: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return empty arrays as a last resort
        return np.array([]), np.array([])

def get_genre_based_recommendations(book_id, source_genres, all_book_genres, num_recommendations=20):
    """
    Get recommendations based on genre matching
    
    Args:
        book_id (str): ID of the source book
        source_genres (dict): Genre data for the source book
        all_book_genres (dict): Genre data for all books
        num_recommendations (int): Number of recommendations to return
        
    Returns:
        list: List of (book_id, score) tuples
    """
    if not source_genres:
        return []
    
    # Calculate genre similarity scores for all books
    book_scores = []
    
    # Get the total genre weight for source book
    source_total = sum(source_genres.values())
    
    # Safety check for zero weights in source genres
    if source_total == 0:
        print(f"Warning: Book {book_id} has genres with zero total weight")
        return []
    
    # Source book genre preferences (normalized)
    source_preferences = {genre: weight/source_total for genre, weight in source_genres.items()}
    print(f"Source book has genres: {list(source_preferences.keys())}")
    
    # Calculate a unique jitter factor based on book_id
    # This will ensure different books get different recommendations
    book_id_hash = hash(book_id) % 2147483647
    rng = np.random.RandomState(book_id_hash)
    
    # Process each book in the dataset
    for candidate_id, candidate_genres in all_book_genres.items():
        # Skip the source book itself
        if candidate_id == book_id:
            continue
            
        # Skip books with no genre data or empty genres
        if not candidate_genres or sum(candidate_genres.values()) == 0:
            continue
            
        # Calculate similarity score
        try:
            similarity = calculate_genre_similarity(source_preferences, candidate_genres)
            
            # Add jitter based on book_id hash
            # This ensures different source books will get different ordering
            jitter = rng.uniform(0.85, 1.15)  # ±15% variation
            
            # Adjust similarity with jitter
            adjusted_similarity = similarity * jitter
            
            book_scores.append((candidate_id, adjusted_similarity))
        except Exception as e:
            print(f"Error calculating similarity for book {candidate_id}: {e}")
            continue
    
    # Sort by similarity score (descending)
    book_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N recommendations
    return book_scores[:num_recommendations]

def calculate_genre_similarity(source_preferences, candidate_genres):
    """
    Calculate the similarity between two books based on their genres
    
    Args:
        source_preferences (dict): Normalized genre preferences for source book
        candidate_genres (dict): Genre data for candidate book
        
    Returns:
        float: Similarity score (0-1)
    """
    # Normalize candidate genres
    candidate_total = sum(candidate_genres.values())
    
    # Check for zero total to avoid division by zero
    if candidate_total == 0:
        return 0.1  # Return low base similarity for books with zero genre weights
        
    candidate_preferences = {genre: weight/candidate_total for genre, weight in candidate_genres.items()}
    
    # Find common genres
    common_genres = set(source_preferences.keys()) & set(candidate_preferences.keys())
    
    # If no common genres, return low base similarity
    if not common_genres:
        return 0.1  # Small non-zero value to allow some diversity
    
    # Calculate weighted similarity
    similarity = 0
    for genre in common_genres:
        # Multiply the weights from both books for this genre
        genre_similarity = source_preferences[genre] * candidate_preferences[genre]
        similarity += genre_similarity
    
    # Normalize similarity (0-1)
    # More common genres will result in higher similarity
    similarity = min(1.0, similarity * (1 + 0.1 * len(common_genres)))
    
    return similarity

def load_book_data():
    """Load and prepare book data"""
    # Load book ID mapping
    book_map = pd.read_csv('data/book_id_map.csv')
    # Load detailed book information
    books_df = pd.read_csv('data/preprocessed_books.csv')
    # Merge the dataframes
    merged_df = pd.merge(book_map, books_df, on='book_id', how='inner')
    return merged_df

def print_book_info(book_id, books_df, q_value=None, source_book=None):
    """Print book ID and similarity metrics"""
    try:
        book_info = books_df[books_df['book_id'] == book_id].iloc[0]
        
        print(f"\nBook ID: {book_id}")
        print("-" * 30)
        
        # Book Stats
        rating = float(book_info['average_rating']) if pd.notna(book_info['average_rating']) else 0
        print(f"Rating: {rating:.2f}")
        
        # Rating similarity
        if source_book is not None:
            try:
                source_rating = float(source_book['average_rating']) if pd.notna(source_book['average_rating']) else 0
                print(f"Source Book Rating: {source_rating:.2f}")
                if rating > 0 and source_rating > 0:
                    rating_sim = 1 - abs(rating - source_rating) / 5.0
                    print(f"Rating Similarity: {rating_sim:.2%}")
            except (ValueError, KeyError):
                print("Rating Similarity: N/A")
        
        # DQN score
        if q_value is not None:
            normalized_q = 1 / (1 + np.exp(-q_value))  # Sigmoid normalization
            print(f"DQN Score: {normalized_q:.2%}")
        
        print("-" * 30)
        
    except Exception as e:
        print(f"Error calculating similarities: {str(e)}")

def calculate_confidence_scores(q_values, temperature=0.1):
    """Calculate confidence scores with temperature scaling"""
    # Scale Q-values to make differences more pronounced
    scaled_q = q_values / temperature
    # Apply softmax with scaling
    exp_q = np.exp(scaled_q - np.max(scaled_q))  # Subtract max for numerical stability
    return exp_q / exp_q.sum()

def main(book_id=None):
    try:
        # Load book data
        books_df = load_book_data()
        book_ids = set(books_df['book_id'].values)
        
        # Get source book - either use provided book_id or choose a random one
        if book_id is not None:
            source_id = book_id
            print(f"\nSource Book ID: {source_id}")
        else:
            # Get random source book
            random_book = books_df.sample(n=1).iloc[0]
            source_id = random_book['book_id']
            print(f"\nSource Book ID: {source_id}")
        
        # Get recommendations
        recommendations, q_values = get_book_recommendations(source_id)
        
        if recommendations is not None and len(recommendations) > 0:
            print("\nTop 10 Similar Books:")
            
            valid_recommendations = []
            for rec_id, q_val in zip(recommendations, q_values):
                if rec_id != source_id and rec_id in book_ids:
                    valid_recommendations.append((rec_id, q_val))
                if len(valid_recommendations) >= 10:
                    break
            
            if not valid_recommendations:
                print("No valid recommendations found")
                return []
            
            # Display recommendations with similarity scores
            for i, (rec_id, q_val) in enumerate(valid_recommendations, 1):
                print(f"\nRecommendation #{i}")
                print_book_info(rec_id, books_df, q_val, 
                                books_df[books_df['book_id'] == source_id].iloc[0] if source_id in books_df['book_id'].values else None)
            
            # Return the recommendation IDs
            return [rec_id for rec_id, _ in valid_recommendations]
                
        else:
            print("No recommendations generated")
            return []
    
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def get_recommendations(book_id):
    """Get recommendations for a given book ID to be used by the API
    
    Args:
        book_id (str): The ID of the book to get recommendations for
        
    Returns:
        list: A list of recommended book IDs
    """
    try:
        print(f"\nGenerating recommendations for book {book_id}")
        
        # Use existing function to get recommendations
        recommendations, q_values = get_book_recommendations(book_id)
        
        if recommendations is None or len(recommendations) == 0:
            print(f"No recommendations found for book {book_id}")
            return []
            
        # Load book data to validate recommendations
        try:
            books_df = load_book_data()
            book_ids = set(books_df['book_id'].values)
            print(f"Loaded {len(book_ids)} valid book IDs for validation")
        except Exception as e:
            print(f"Warning: Could not load book data for validation: {e}")
            # If we can't load book data, just return the recommendations without validation
            print(f"Returning unvalidated recommendations: {recommendations[:10]}")
            return [str(rec_id) for rec_id in recommendations[:10]]
        
        # Filter out invalid recommendations and the source book itself
        valid_recommendations = []
        for rec_id in recommendations:
            # Convert to string for API compatibility
            str_rec_id = str(rec_id)
            # Only add if it's not the source book and is a valid book ID
            if str_rec_id != book_id and (len(book_ids) == 0 or rec_id in book_ids):
                valid_recommendations.append(str_rec_id)
                print(f"Added valid recommendation: {str_rec_id}")
            # Stop when we have 10 valid recommendations
            if len(valid_recommendations) >= 10:
                break
        
        print(f"Returning {len(valid_recommendations)} valid recommendations for book {book_id}")
        return valid_recommendations
    
    except Exception as e:
        print(f"Error getting recommendations for book {book_id}: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    main()