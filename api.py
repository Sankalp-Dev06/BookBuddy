from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId
import uuid
import hashlib
from recommend_for_book import main as get_recommendations
from recommend_for_user import get_recommendations_for_logged_user
import json
import csv
import time
from recommend_for_book import get_recommendations as get_book_recommendations

app = Flask(__name__)
CORS(app)  # Enable CORS

client = MongoClient('mongodb://localhost:27017/')
db = client["bookbuddy"]
collection = db["books"]
users_collection = db["Users"]

# Request cache to prevent duplicate processing
recommendation_cache = {}
book_detail_cache = {}  # Add a cache for book details
cache_timeout = 400  # Cache timeout in seconds
# Add a cache for API responses (full JSON responses)
response_cache = {}

@app.route('/api/search', methods=['GET'])
def search_books():
    query = request.args.get('q', '')
    if not query:
        return jsonify([])
    
    try:
        pattern = f".*{query}.*"
        results = collection.find({
            "$or": [
                {"title": {"$regex": pattern, "$options": "i"}},
                {"author": {"$regex": pattern, "$options": "i"}}
            ]
        }).limit(20)
        
        books = []
        for book in results:
            try:
                rating = float(book.get("average_rating", 0))
            except (ValueError, TypeError):
                rating = 0.0
                
            books.append({
                "book_id": str(book.get("book_id", "")),
                "title": book.get("title", ""),
                "author": book.get("author", "Unknown"),
                "image_url": book.get("image_url", ""),
                "average_rating": rating,
                "genres": book.get("genres", [])
            })
        
        return jsonify(books)
    
    except Exception as e:
        print(f"Search error: {str(e)}")
        return jsonify({"error": "Search failed"}), 500

@app.route('/api/books', methods=['GET'])
def get_books():
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        selected_genre = request.args.get('genre', '')

        # Build query based on genre filter
        query = {}
        if selected_genre:
            # Match books where the genre exists either as a key in the genres dictionary
            # or as an element in the genres array
            query['$or'] = [
                {'genres.' + selected_genre: {'$exists': True}},  # For dictionary format
                {'genres': selected_genre}  # For array format
            ]

        # Get total count for pagination with genre filter
        total_books = collection.count_documents(query)
        total_pages = (total_books + per_page - 1) // per_page

        # Calculate skip value for pagination
        skip = (page - 1) * per_page

        # Get books with pagination and genre filter
        books = list(collection.find(query)
                    .skip(skip)
                    .limit(per_page))
        
        formatted_books = []
        for book in books:
            try:
                rating = float(book.get("average_rating", 0))
            except (ValueError, TypeError):
                rating = 0.0
                
            # Get genres list - handle both dict and list formats
            genres = book.get("genres", {})
            if isinstance(genres, dict):
                genres = list(genres.keys())
            elif not isinstance(genres, list):
                genres = []
                
            formatted_books.append({
                "id": str(book.get("book_id", "")),
                "title": book.get("title", ""),
                "author": book.get("author", "Unknown"),
                "rating": rating,
                "coverUrl": book.get("image_url", ""),
                "description": book.get("description", ""),
                "genres": genres
            })
        
        return jsonify({
            "books": formatted_books,
            "pagination": {
                "current_page": page,
                "per_page": per_page,
                "total_books": total_books,
                "total_pages": total_pages
            }
        })
        
    except Exception as e:
        print(f"Error fetching books: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Failed to fetch books",
            "message": str(e)
        }), 500

@app.route('/api/books/<book_id>', methods=['GET'])
def get_book_details(book_id):
    try:
        book = collection.find_one({"book_id": book_id})
        if not book:
            return jsonify({"error": "Book not found"}), 404

        # Convert genres object to simple array of genre names
        genres_obj = book.get("genres", {})
        genres_array = list(genres_obj.keys()) if isinstance(genres_obj, dict) else []

        # Format the book data
        book_data = {
            "book_id": str(book.get("book_id", "")),
            "title": book.get("title", ""),
            "author": book.get("author", "Unknown"),
            "description": book.get("description", "No description available."),
            "image_url": book.get("image_url", ""),
            "average_rating": float(book.get("average_rating", 0)),
            "genres": genres_array  # Now a simple array of genre names
        }
        
        return jsonify(book_data)
        
    except Exception as e:
        print(f"Error fetching book details: {str(e)}")
        return jsonify({"error": "Failed to fetch book details"}), 500

@app.route('/api/genres', methods=['GET'])
def get_genres():
    try:
        # Initialize an empty set to store unique genres
        unique_genres = set()
        
        # Get all books and process their genres
        books = collection.find({}, {"genres": 1})
        for book in books:
            genres = book.get("genres", {})
            # Handle both dictionary and list formats
            if isinstance(genres, dict):
                unique_genres.update(genres.keys())
            elif isinstance(genres, list):
                unique_genres.update(genres)
            
        # Convert set to sorted list and filter out any empty strings or None values
        genres_list = sorted([genre for genre in unique_genres if genre and isinstance(genre, str)])
        
        # Return a simple array of genres instead of an object
        return jsonify(genres_list)
        
    except Exception as e:
        print(f"Error fetching genres: {str(e)}")
        return jsonify([]), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return jsonify({"error": "Missing credentials"}), 400

        # Find user in database
        user = db.Users.find_one({"username": username})
        
        if user and user['password'] == password:
            # Don't send password in response
            user_data = {
                "username": user['username'],
                "user_id": user['user_id']
            }
            return jsonify({"success": True, "user": user_data})
        else:
            return jsonify({"error": "Invalid credentials"}), 401

    except Exception as e:
        print(f"Login error: {str(e)}")
        return jsonify({"error": "Login failed"}), 500

@app.route('/api/profile', methods=['GET'])
def get_profile():
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({"error": "User ID required"}), 400

        user = db.Users.find_one({"user_id": user_id})
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Don't send password in response
        user_data = {
            "username": user['username'],
            "user_id": user['user_id']
        }
        return jsonify(user_data)

    except Exception as e:
        print(f"Profile error: {str(e)}")
        return jsonify({"error": "Failed to fetch profile"}), 500
    
    
@app.route('/api/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        email = data.get('email')

        if not username or not password or not email:
            return jsonify({"error": "Missing required fields"}), 400

        # Check if username already exists
        if users_collection.find_one({"username": username}):
            return jsonify({"error": "Username already exists"}), 409

        # Check if email already exists
        if users_collection.find_one({"email": email}):
            return jsonify({"error": "Email already exists"}), 409

        # Generate a unique user ID using uuid and hash it
        unique_string = str(uuid.uuid4()) + username
        user_id = hashlib.md5(unique_string.encode()).hexdigest()

        # Create new user document with only essential fields
        new_user = {
            "user_id": user_id,
            "username": username,
            "password": password,  # In production, hash this password!
            "email": email
        }

        # Insert the new user
        result = users_collection.insert_one(new_user)

        if result.inserted_id:
            return jsonify({
                "success": True,
                "user": {
                    "username": username,
                    "user_id": user_id,
                    "email": email
                }
            })
        else:
            return jsonify({"error": "Failed to create user"}), 500

    except Exception as e:
        print(f"Signup error: {str(e)}")
        return jsonify({"error": "Registration failed"}), 500
    
@app.route('/api/books/<book_id>/recommendations', methods=['GET'])
def get_book_recommendations_api(book_id):
    """Get similar book recommendations for a specific book"""
    try:
        # Get number of recommendations from query parameters (default to 5)
        num_recommendations = request.args.get('limit', 5, type=int)
        
        # Log the request
        print(f"\nProcessing recommendation request for book: {book_id}")
        
        # Get recommendations from the recommend_for_book module
        recommendations = get_book_recommendations(book_id)
        
        if not recommendations:
            # If no recommendations are found, return an empty result
            print(f"No recommendations found for book {book_id}")
            return jsonify({
                'status': 'success',
                'message': 'No recommendations found for this book',
                'recommendations': []
            })
        
        # Limit to requested number
        recommendations = recommendations[:num_recommendations]
        print(f"Retrieved {len(recommendations)} raw recommendations for book {book_id}")
        print(f"Recommendation IDs for book {book_id}: {recommendations}")
        
        # Fetch full book details from MongoDB for each recommendation
        book_collection = collection
        recommended_books = []
        
        for rec_id in recommendations:
            try:
                book = book_collection.find_one({'book_id': rec_id})
                if book:
                    # Convert ObjectId to string for JSON serialization
                    book['_id'] = str(book['_id'])
                    recommended_books.append(book)
                    print(f"Added book {rec_id} to recommendations")
                else:
                    print(f"Book {rec_id} not found in database")
            except Exception as e:
                print(f"Error fetching details for book {rec_id}: {str(e)}")
        
        print(f"Returning {len(recommended_books)} recommendations for book {book_id}")
        
        # Return recommendations with proper structure
        return jsonify({
            'status': 'success',
            'message': f'Found {len(recommended_books)} recommendations',
            'recommendations': recommended_books
        })
        
    except Exception as e:
        print(f"Error getting book recommendations: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': 'Failed to get recommendations',
            'error': str(e)
        }), 500

@app.route('/api/recommendations/featured', methods=['GET'])
def get_featured_recommendations():
    try:
        print("\n--- Starting Featured Recommendations API Call ---")
        user_data = request.args.get('userData')
        
        if not user_data:
            print("No userData provided in request")
            return jsonify({'books': []})
        
        print(f"Received userData: {user_data}")
        user_dict = json.loads(user_data)
        # Check for both possible user ID keys (userId and user_id)
        user_id = user_dict.get('userId') or user_dict.get('user_id')
        
        if not user_id:
            print("No user ID provided in request")
            return jsonify({'books': []})
        
        print(f"Processing request for user: {user_id}")
        
        # Debug user ID
        print(f"User ID type: {type(user_id)}, Length: {len(user_id)}")
        print(f"Full user ID: {user_id}")
        
        # Try to directly scan the user_book.csv file first (most reliable method)
        try:
            # For compatibility - check if ID is in the database format (32 chars)
            db_format_id = user_id
            if len(user_id) > 32:
                print(f"User ID is longer than 32 chars, truncating for database lookup")
                db_format_id = user_id[:32]
                print(f"Truncated to: {db_format_id}")
            
            # First, try exact matching with the full user ID
            print(f"Trying exact ID match with: {db_format_id}")
            
            # Check for exact match in user_id_map.csv
            try:
                print("Checking user_id_map.csv for user ID mapping...")
                user_id_matches = []
                with open('data/user_id_map.csv', 'r', encoding='utf-8') as map_file:
                    map_reader = csv.reader(map_file)
                    # Skip header
                    next(map_reader)
                    for row in map_reader:
                        if len(row) >= 2 and (row[1] == user_id or row[1].startswith(db_format_id)):
                            user_id_matches.append(row)
                
                if user_id_matches:
                    print(f"Found {len(user_id_matches)} matching user IDs in mapping:")
                    for match in user_id_matches:
                        print(f"  Mapped ID: {match}")
            except Exception as map_err:
                print(f"Error checking user ID mapping: {str(map_err)}")
            
            book_ids = []
            matched_lines = 0
            csv_file_path = 'data/user_book.csv'
            
            print(f"Opening file: {csv_file_path}")
            with open(csv_file_path, 'r', encoding='utf-8') as file:
                # Skip header
                next(file)
                
                # Scan for lines starting with the user prefix
                for line in file:
                    if line.startswith(db_format_id):
                        matched_lines += 1
                        try:
                            # The format is: userID bookID rating
                            parts = line.strip().split(' ')
                            if len(parts) >= 2:
                                book_id = parts[1]
                                book_ids.append(book_id)
                                print(f"Found match: {line.strip()} -> book_id: {book_id}")
                        except Exception as e:
                            print(f"Error parsing line: {line}, Error: {str(e)}")

            # Using CSV reader for proper parsing
            try:
                with open(csv_file_path, 'r', encoding='utf-8') as file:
                    # Create CSV reader
                    csv_reader = csv.reader(file)
                    
                    # Skip header
                    next(csv_reader)
                    
                    # Scan through each row
                    for row in csv_reader:
                        # Check if we have enough columns and user ID matches
                        if len(row) >= 2 and row[0].startswith(db_format_id):
                            matched_lines += 1
                            book_id = row[1]
                            book_ids.append(book_id)
                            print(f"Found match: {','.join(row)} -> book_id: {book_id}")
            except Exception as e:
                print(f"Error reading CSV file: {str(e)}")
                import traceback
                traceback.print_exc()
                
            # If we didn't find any matches, try using the numeric user ID approach
            if not book_ids:
                print("No matches found using user ID prefix. Trying numeric user ID approach...")
                try:
                    # Find numeric user ID in mapping
                    numeric_user_id = None
                    with open('data/user_id_map.csv', 'r', encoding='utf-8') as map_file:
                        map_reader = csv.reader(map_file)
                        # Skip header
                        next(map_reader)
                        for row in map_reader:
                            if len(row) >= 2 and row[1] == user_id:
                                numeric_user_id = row[0]
                                print(f"Found numeric user ID: {numeric_user_id}")
                                break
                    
                    # If we found a numeric ID, look for interactions with that ID
                    if numeric_user_id:
                        print(f"Looking for interactions with numeric user ID: {numeric_user_id}")
                        with open(csv_file_path, 'r', encoding='utf-8') as file:
                            csv_reader = csv.reader(file)
                            # Skip header
                            next(csv_reader)
                            
                            # Scan through each row
                            for row in csv_reader:
                                # Check if numeric ID matches
                                if len(row) >= 2 and row[0] == numeric_user_id:
                                    matched_lines += 1
                                    book_id = row[1]
                                    book_ids.append(book_id)
                                    print(f"Found match with numeric ID: {','.join(row)} -> book_id: {book_id}")
                        
                        print(f"Found {matched_lines} matches using numeric user ID approach")
                    else:
                        print("Could not find numeric user ID in mapping")
                        
                except Exception as e:
                    print(f"Error in numeric user ID approach: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            print(f"Found {matched_lines} matches for user {db_format_id}")
            print(f"Extracted {len(book_ids)} book IDs")
            
            # If we found books, get the top 20
            if book_ids:
                book_ids = book_ids[:20]  # Limit to 20 books
                
                # Load book details
                books_data = []
                with open('data/preprocessed_books.csv', 'r', encoding='utf-8') as book_file:
                    reader = csv.DictReader(book_file)
                    book_dict = {row['id']: row for row in reader}
                
                for book_id in book_ids:
                    if book_id in book_dict:
                        book = book_dict[book_id]
                        books_data.append({
                            'book_id': book_id,
                            'title': book['title'],
                            'author': book['authors'],
                            'image_url': book['image_url'],
                            'average_rating': book['average_rating']
                        })
                 
                for book_id in book_ids:
                    if book_id in book_dict:
                        book = book_dict[book_id]
                        # Handle different column names in CSV
                        books_data.append({
                            'book_id': book_id,
                            'title': book.get('title', ''),
                            'author': book.get('authors', book.get('author', 'Unknown')),
                            'image_url': book.get('image_url', ''),
                            'average_rating': book.get('average_rating', book.get('rating', '0'))
                        })
                 
                for book_id in book_ids:
                    if book_id in book_dict:
                        book = book_dict[book_id]
                        # Handle different column names in CSV
                        books_data.append({
                            'book_id': book_id,
                            'title': book.get('title', ''),
                            'author': book.get('authors', book.get('author', 'Unknown')),
                            'image_url': book.get('image_url', ''),
                            'average_rating': book.get('average_rating', book.get('rating', '0'))
                        })

            # If we found books, get the top 20
            if book_ids:
                book_ids = book_ids[:20]  # Limit to 20 books
                
                # We need to fetch additional book details from the database
                # since preprocessed_books.csv only has IDs and ratings
                books_data = []
                
                # First connect to MongoDB for complete book data
                try:
                    for book_id in book_ids:
                        # Try to find the book in the MongoDB collection
                        book = collection.find_one({"book_id": str(book_id)})
                        if book:
                            books_data.append({
                                'book_id': book_id,
                                'title': book.get('title', 'Unknown Title'),
                                'author': book.get('author', 'Unknown Author'),
                                'image_url': book.get('image_url', ''),
                                'average_rating': float(book.get('average_rating', 0))
                            })
                except Exception as e:
                    print(f"Error fetching from database: {str(e)}")
                
                # If we couldn't find books in MongoDB, try popular books
                if not books_data:
                    print("No books found in database, using popular books")
                    try:
                        # Get books from the regular API
                        popular_books = list(collection.find({}).limit(20))
                        for book in popular_books:
                            books_data.append({
                                'book_id': book.get('book_id', ''),
                                'title': book.get('title', 'Unknown Title'),
                                'author': book.get('author', 'Unknown Author'),
                                'image_url': book.get('image_url', ''),
                                'average_rating': float(book.get('average_rating', 0))
                            })
                    except Exception as e:
                        print(f"Error getting popular books: {str(e)}")
                
                print(f"Returning {len(books_data)} book recommendations")
                return jsonify({'books': books_data})
        except Exception as e:
            print(f"Error while directly scanning user_book.csv: {str(e)}")
            import traceback
            traceback.print_exc()
            
        # Fallback: Try using the recommendation engine
        try:
            print("Attempting to use recommendation engine...")
            from recommend_for_user import get_recommendations_for_logged_user
            recommended_ids = get_recommendations_for_logged_user(user_id)
            
            if recommended_ids:
                books_data = []
                with open('data/preprocessed_books.csv', 'r', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    book_dict = {row['id']: row for row in reader}
                
                for book_id in recommended_ids:
                    if book_id in book_dict:
                        book = book_dict[book_id]
                        books_data.append({
                            'book_id': book_id,
                            'title': book['title'],
                            'author': book['authors'],
                            'image_url': book['image_url'],
                            'average_rating': book['average_rating']
                        })

            if recommended_ids:
                books_data = []
                
                # Get book details from MongoDB
                try:
                    for book_id in recommended_ids[:20]:  # Limit to 20 books
                        # Try to find the book in collection
                        book = collection.find_one({"book_id": str(book_id)})
                        if book:
                            books_data.append({
                                'book_id': book_id,
                                'title': book.get('title', 'Unknown Title'),
                                'author': book.get('author', 'Unknown Author'),
                                'image_url': book.get('image_url', ''),
                                'average_rating': float(book.get('average_rating', 0))
                            })
                except Exception as e:
                    print(f"Error fetching from MongoDB: {str(e)}")
                    
                # If no books found, try getting popular books
                if not books_data:
                    print("No recommended books found in database, using popular books")
                    try:
                        popular_books = list(collection.find({}).limit(20))
                        for book in popular_books:
                            books_data.append({
                                'book_id': book.get('book_id', ''),
                                'title': book.get('title', 'Unknown Title'),
                                'author': book.get('author', 'Unknown Author'),
                                'image_url': book.get('image_url', ''), 
                                'average_rating': float(book.get('average_rating', 0))
                            })
                    except Exception as e:
                        print(f"Error getting popular books: {str(e)}")
                
                print(f"Returning {len(books_data)} book recommendations from engine")
                return jsonify({'books': books_data})
        except Exception as e:
            print(f"Error using recommendation engine: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # If we get here, return empty array
        print("No recommendations found for user")
        return jsonify({'books': []})

        # If we get here, try popular books from MongoDB
        print("No user-specific recommendations found, trying popular books")
        try:
            popular_books = list(collection.find({}).limit(20))
            books_data = []
            
            for book in popular_books:
                books_data.append({
                    'book_id': book.get('book_id', ''),
                    'title': book.get('title', 'Unknown Title'),
                    'author': book.get('author', 'Unknown Author'),
                    'image_url': book.get('image_url', ''), 
                    'average_rating': float(book.get('average_rating', 0))
                })
                
            print(f"Returning {len(books_data)} popular books")
            return jsonify({'books': books_data})
        except Exception as e:
            print(f"Error getting popular books: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # If all else fails, return empty array
        print("No recommendations found at all")
        return jsonify({'books': []})
        
    except Exception as e:
        print(f"Error in featured recommendations endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'books': [], 'error': str(e)})

@app.route('/api/user/<user_id>/recommendations', methods=['GET'])
def get_user_recommendations(user_id):
    # Check if this request is in cache and still valid
    current_time = time.time()
    cache_key = f"user_rec_{user_id}"
    
    # Check if we have a complete cached response
    if cache_key in response_cache:
        cache_entry = response_cache[cache_key]
        if current_time - cache_entry['timestamp'] < cache_timeout:
            print(f"Returning complete cached response for user {user_id}")
            return cache_entry['response']
    
    # Check if we have cached recommendation IDs
    if cache_key in recommendation_cache:
        cache_entry = recommendation_cache[cache_key]
        # If cache entry is still valid (not expired)
        if current_time - cache_entry['timestamp'] < cache_timeout:
            print(f"Returning cached recommendations for user {user_id}")
            recommended_ids = cache_entry['recommended_ids']
            if not recommended_ids or len(recommended_ids) == 0:
                # Return cached fallback response
                if 'fallback_response' in cache_entry:
                    return cache_entry['fallback_response']
                # Generate fallback if not cached
                response = get_popular_books_fallback("No recommendations found for user")
                # Update cache with fallback response
                recommendation_cache[cache_key]['fallback_response'] = response
                return response
            # Continue with fetching book details for the cached recommendation IDs
        else:
            # Clear expired cache entry
            del recommendation_cache[cache_key]
    
    try:
        # Validate user ID
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400
        
        # Use a simplified log separator to reduce log verbosity
        print(f"\nProcessing recommendations for user: {user_id}")
            
        # Get recommendations for user
        recommended_ids = None
        try:
            recommended_ids = get_recommendations_for_logged_user(user_id)
            print(f"Received {len(recommended_ids) if recommended_ids else 0} recommendation IDs")
        except Exception as rec_error:
            print(f"Error getting recommendations from recommendation engine: {rec_error}")
            import traceback
            traceback.print_exc()
        
        if not recommended_ids or len(recommended_ids) == 0:
            response = get_popular_books_fallback("No recommendations found for user")
            # Cache both the recommendation IDs and the fallback response
            recommendation_cache[cache_key] = {
                'recommended_ids': recommended_ids,
                'fallback_response': response,
                'timestamp': current_time
            }
            # Cache the complete response
            response_cache[cache_key] = {
                'response': response,
                'timestamp': current_time
            }
            return response
            
        # Cache the recommendation IDs
        recommendation_cache[cache_key] = {
            'recommended_ids': recommended_ids,
            'timestamp': current_time
        }
        
        # Load book ID mapping to translate between recommendation IDs and MongoDB IDs
        book_id_map = {}
        try:
            # Load the book_id_map.csv file once and cache it for future use
            import csv
            if not hasattr(app, 'book_id_map'):
                app.book_id_map = {}
                with open('data/book_id_map.csv', 'r', encoding='utf-8') as map_file:
                    reader = csv.reader(map_file)
                    next(reader)  # Skip header
                    for row in reader:
                        if len(row) >= 2:
                            csv_id = row[0]
                            real_id = row[1]
                            app.book_id_map[csv_id] = real_id
                print(f"Loaded {len(app.book_id_map)} entries from book_id_map.csv")
            book_id_map = app.book_id_map
        except Exception as e:
            print(f"Error loading book ID mapping: {e}")
        
        # Fetch book details for recommendations
        recommended_books = []
        found_count = 0
        not_found_count = 0
        
        for book_id in recommended_ids[:20]:  # Limit to 20 recommendations
            # Skip if already in user's history
            
            # Try to find MongoDB ID from our mapping
            mongo_id = str(book_id)
            if str(book_id) in book_id_map:
                mongo_id = book_id_map[str(book_id)]
            
            # Check if book details are cached
            book_cache_key = f"book_{mongo_id}"
            if book_cache_key in book_detail_cache:
                book_data = book_detail_cache[book_cache_key]
                recommended_books.append(book_data)
                found_count += 1
                continue
            
            # Try to find the book in collection
            book = collection.find_one({"book_id": mongo_id})
            if not book:
                # Try without mapping (raw ID)
                book = collection.find_one({"book_id": str(book_id)})
            
            if book:
                # Format book data for API response
                try:
                    book_data = {
                        'book_id': str(book.get("book_id", "")),
                        'title': book.get('title', 'Unknown Title'),
                        'author': book.get('author', 'Unknown Author'),
                        'image_url': book.get('image_url', ''),
                        'average_rating': float(book.get('average_rating', 0))
                    }
                    
                    # Cache the book details
                    book_detail_cache[book_cache_key] = book_data
                    
                    recommended_books.append(book_data)
                    found_count += 1
                except Exception as format_error:
                    print(f"Error formatting book data: {format_error}")
                    not_found_count += 1
            else:
                not_found_count += 1
                
        print(f"Found {found_count} books and missed {not_found_count} books")
        
        if recommended_books:
            print(f"Returning {len(recommended_books)} books from recommendations")
            response = jsonify(recommended_books)
            # Cache the complete response
            response_cache[cache_key] = {
                'response': response,
                'timestamp': current_time
            }
            return response
        
        # If no books were found via recommendations, try popular books as fallback
        response = get_popular_books_fallback("No recommended books found in database")
        # Cache both the recommendation IDs and the fallback response
        recommendation_cache[cache_key]['fallback_response'] = response
        # Cache the complete response
        response_cache[cache_key] = {
            'response': response,
            'timestamp': current_time
        }
        return response
        
    except Exception as e:
        print(f"Error getting user recommendations: {str(e)}")
        import traceback
        traceback.print_exc()
        error_response = jsonify({"error": "Failed to get recommendations"}), 500
        return error_response

# Helper function to get popular books as fallback
def get_popular_books_fallback(reason):
    # Check if popular books are already cached
    cache_key = "popular_books"
    current_time = time.time()
    
    if cache_key in response_cache:
        cache_entry = response_cache[cache_key]
        if current_time - cache_entry['timestamp'] < cache_timeout:
            print(f"Using cached popular books fallback")
            return cache_entry['response']
    
    print(f"Using popular books fallback. Reason: {reason}")
    try:
        popular_books = list(collection.find({}).limit(20))
        fallback_books = []
        
        for book in popular_books:
            fallback_books.append({
                "book_id": book.get("book_id", ""),
                "title": book.get("title", "Unknown Title"),
                "author": book.get("author", "Unknown Author"),
                "image_url": book.get("image_url", ""),
                "average_rating": float(book.get("average_rating", 0))
            })
        
        print(f"Returning {len(fallback_books)} popular books as fallback")
        response = jsonify(fallback_books)
        
        # Cache the popular books response
        response_cache[cache_key] = {
            'response': response,
            'timestamp': current_time
        }
        
        return response
    except Exception as e:
        print(f"Error getting popular books fallback: {e}")
        return jsonify([])

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)