function handleApiError(error, elementId) {
    console.error('API Error:', error);
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `
            <div class="error-message">
                <i class="fa-solid fa-exclamation-circle"></i>
                <p>Failed to load data. Please try again later.</p>
            </div>
        `;
    }
}

// Add this function at the beginning of the file
async function getBooks() {
    try {
        // Call your API endpoint to get books
        const response = await fetch(`${API_BASE_URL}/api/books`);
        if (!response.ok) {
            throw new Error('Failed to fetch books');
        }
        const books = await response.json();
        return books;
    } catch (error) {
        console.error('Error fetching books:', error);
        return [];
    }
}

// Add this at the beginning of your file
document.addEventListener('DOMContentLoaded', () => {
    initializeNavigation();
    loadHomePage();
});

function initializeNavigation() {
    const navLinks = document.querySelectorAll('.nav-link[data-page]');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Remove active class from all links
            navLinks.forEach(l => l.classList.remove('active'));
            
            // Add active class to clicked link
            link.classList.add('active');
            
            // Get the page id from data-page attribute
            const pageId = link.getAttribute('data-page');
            
            // Hide all pages
            document.querySelectorAll('.page').forEach(page => {
                page.classList.remove('active');
            });
            
            // Show the selected page
            const selectedPage = document.getElementById(`${pageId}-page`);
            if (selectedPage) {
                selectedPage.classList.add('active');
                
                // Load page-specific content if needed
                switch(pageId) {
                    case 'home':
                        loadHomePage();
                        break;
                    case 'explore':
                        loadExplorePage();
                        break;
                    case 'search':
                        loadSearchPage();
                        break;
                    case 'about':
                        loadAboutPage();
                        break;
                }
            }
        });
    });

    // Handle profile navigation separately
    const profileLink = document.getElementById('profile-nav-link');
    if (profileLink) {
        profileLink.addEventListener('click', (e) => {
            e.preventDefault();
            const isLoggedIn = localStorage.getItem('isLoggedIn') === 'true';
            
            if (!isLoggedIn) {
                window.location.href = 'login.html';
            } else {
                // Remove active class from all links
                navLinks.forEach(l => l.classList.remove('active'));
                profileLink.classList.add('active');
                
                // Show profile page
                document.querySelectorAll('.page').forEach(page => {
                    page.classList.remove('active');
                });
                const profilePage = document.getElementById('profile-page');
                if (profilePage) {
                    profilePage.classList.add('active');
                    loadProfilePage();
                }
            }
        });
    }
}

async function loadExplorePage() {
    // Get required elements
    const elements = getRequiredElements([
        'explore-books-grid',
        'pagination',
        'genre-filter' // Add this element
    ]);

    if (!elements) {
        console.error('Cannot initialize explore page - missing elements');
        return;
    }

    // Load genres for the filter
    try {
        const response = await fetch(`${API_BASE_URL}/api/genres`);
        if (!response.ok) throw new Error('Failed to fetch genres');
        
        const genres = await response.json();
        
        // Populate genre filter
        elements['genre-filter'].innerHTML = `
            <option value="">All Genres</option>
            ${genres.map(genre => `
                <option value="${genre}">${genre}</option>
            `).join('')}
        `;

        // Add event listener for genre filter
        elements['genre-filter'].addEventListener('change', async (e) => {
            const selectedGenre = e.target.value;
            await loadBooks(1, selectedGenre);
        });

    } catch (error) {
        console.error('Error loading genres:', error);
        elements['genre-filter'].innerHTML = '<option value="">All Genres</option>';
    }

    // Load initial books
    await loadBooks(1);
}

function loadSearchPage() {
    const searchPage = document.getElementById('search-page');
    if (searchPage) {
        // Add your search page loading logic here
        console.log('Loading search page...');
    }
}

function loadAboutPage() {
    const aboutPage = document.getElementById('about-page');
    if (aboutPage) {
        // Add your about page loading logic here
        console.log('Loading about page...');
    }
}

function loadProfilePage() {
    const profilePage = document.getElementById('profile-page');
    if (profilePage) {
        // Add your profile page loading logic here
        console.log('Loading profile page...');
    }
}

// DOM Elements
const pagesContainer = document.querySelector('.pages-container');
const pages = document.querySelectorAll('.page');
const navLinks = document.querySelectorAll('.nav-link');

// State Management
let currentActivePage = 'home';
let currentBookId = null;
let currentBooksPage = 1;

// User data (mock)
const userData = {
  name: 'Alex Johnson',
  email: 'alex.johnson@example.com',
  joinDate: 'January 2024',
  bio: 'Avid reader and book collector. I love mystery novels and historical fiction.',
  avatarUrl: 'placeholder.svg'
};

// Toast notification
const toast = document.getElementById('toast');
const toastTitle = document.getElementById('toast-title');
const toastDescription = document.getElementById('toast-description');
const toastCloseBtn = document.querySelector('.toast-close');

// Initialize the application
document.addEventListener('DOMContentLoaded', async () => {
    try {
        // Load homepage content
        await loadHomePage();
    } catch (error) {
        console.error('Error initializing app:', error);
    }
});

// Initialize navigation
function initNavigation() {
  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const pageId = link.dataset.page;
      navigateTo(pageId);
    });
  });
}

// Setup Event Listeners
function setupEventListeners() {
  // Home page links
  document.getElementById('explore-books-btn').addEventListener('click', () => {
    navigateTo('explore');
  });

  // Back buttons
  document.querySelectorAll('.back-button').forEach(btn => {
    btn.addEventListener('click', () => {
      window.history.back();
    });
  });

  // Search input
  const searchInput = document.getElementById('search-input');
  searchInput.addEventListener('input', debounce((e) => {
    const query = e.target.value.trim();
    if (query.length > 0) {
      const results = searchBooks(query);
      displaySearchResults(results, query);
    } else {
      displaySearchMessage('Start typing to search');
    }
  }, 300));

  // Explore page search
  const exploreSearch = document.getElementById('explore-search');
  exploreSearch.addEventListener('input', debounce((e) => {
    const query = e.target.value.trim();
    const filteredBooks = query ? searchBooks(query) : getBooks(currentBooksPage).books;
    renderBookGrid('explore-books-grid', filteredBooks);
  }, 300));

  // Genre filter
  const genreFilter = document.getElementById('genre-filter');
  genreFilter.addEventListener('change', () => {
    applyFilters();
  });

  // Pagination
  document.getElementById('prev-page').addEventListener('click', () => {
    if (currentBooksPage > 1) {
      currentBooksPage--;
      loadExplorePage();
    }
  });

  document.getElementById('next-page').addEventListener('click', () => {
    const totalBooks = getBooks().total;
    const totalPages = Math.ceil(totalBooks / 12);
    if (currentBooksPage < totalPages) {
      currentBooksPage++;
      loadExplorePage();
    }
  });

  // Tab switching in profile page
  const tabButtons = document.querySelectorAll('.tab-btn');
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const tabId = btn.dataset.tab;
      
      // Update active tab button
      tabButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      
      // Update active tab pane
      document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.classList.remove('active');
      });
      document.getElementById(`${tabId}-tab`).classList.add('active');
    });
  });

  // Close toast
  toastCloseBtn.addEventListener('click', () => {
    hideToast();
  });

  // Close dropdown when clicking outside
  document.addEventListener('click', (e) => {
    const dropdown = document.querySelector('.dropdown');
    if (dropdown && !e.target.closest('.dropdown') && dropdown.classList.contains('active')) {
      dropdown.classList.remove('active');
    }
  });
}

// Update the book card click handler in loadHomePage
async function loadHomePage() {
    const featuredBooksGrid = document.getElementById('featured-books-grid');
    if (!featuredBooksGrid) return;

    try {
        // Clear any existing content and show loading indicator
        featuredBooksGrid.innerHTML = `
            <h1>Recommended For You</h1>
            <p class="subtitle">Based on your reading history and preferences</p>
            <div class="loading">Loading books...</div>
        `;
        
        const userData = localStorage.getItem('userData');
        
        if (!userData) {
            console.log("No user data in localStorage, showing login prompt");
            // Show login prompt for guests
            featuredBooksGrid.innerHTML = `
                <h1>Recommended For You</h1>
                <p class="subtitle">Login to get personalized recommendations</p>
                <div class="login-prompt">
                    <i class="fa-solid fa-user-circle"></i>
                    <h2>Personalized Recommendations</h2>
                    <p>Login to get book recommendations tailored to your interests.</p>
                    <a href="login.html" class="btn btn-primary">Login</a>
                </div>
            `;
            return;
        }

        // Parse user data
        let userDataObj;
        try {
            userDataObj = JSON.parse(userData);
            console.log("Parsed user data:", JSON.stringify(userDataObj));
        } catch (parseError) {
            console.error("Error parsing user data:", parseError);
            featuredBooksGrid.innerHTML = `
                <h1>Recommended For You</h1>
                <p class="subtitle">Based on your reading history and preferences</p>
                <div class="error-message">
                    <i class="fa-solid fa-exclamation-circle"></i>
                    <p>There was a problem with your user data. Please try logging in again.</p>
                    <a href="login.html" class="btn btn-primary">Login Again</a>
                </div>
            `;
            return;
        }
        
        let userId = userDataObj.userId || userDataObj.user_id;
        
        if (!userId) {
            console.error("No user ID found in stored user data:", userDataObj);
            featuredBooksGrid.innerHTML = `
                <h1>Recommended For You</h1>
                <p class="subtitle">Based on your reading history and preferences</p>
                <div class="error-message">
                    <i class="fa-solid fa-exclamation-circle"></i>
                    <p>Failed to load recommendations. Please try logging in again.</p>
                    <a href="login.html" class="btn btn-primary">Login Again</a>
                </div>
            `;
            return;
        }
        
        // Fetch user recommendations
        console.log(`Fetching recommendations for user ID: ${userId}`);
        const formattedUserId = userId;
        
        let response;
        try {
            response = await fetch(`${API_BASE_URL}/api/user/${formattedUserId}/recommendations`);
            console.log(`API response status: ${response.status}`);
        } catch (fetchError) {
            console.error("Network error fetching recommendations:", fetchError);
            return await loadPopularBooks(featuredBooksGrid, "Network error connecting to recommendation service");
        }
        
        // Check for errors
        if (!response.ok) {
            console.warn(`Failed to fetch recommendations with status: ${response.status}. Trying fallback to popular books.`);
            return await loadPopularBooks(featuredBooksGrid, `API error code: ${response.status}`);
        }
        
        let books;
        try {
            books = await response.json();
            console.log(`Received ${books ? books.length : 0} recommendations from API:`, books);
        } catch (jsonError) {
            console.error("Error parsing API response:", jsonError);
            return await loadPopularBooks(featuredBooksGrid, "Error parsing API response");
        }

        // If no recommendations (empty array returned)
        if (!books || books.length === 0) {
            console.log("No personal recommendations found, fetching popular books as fallback");
            return await loadPopularBooks(featuredBooksGrid, "No personal recommendations available");
        }

        // Display recommended books
        const recommendedBooksHTML = books.map(book => {
            // Ensure we have values for all required properties
            const bookId = book.book_id || book.id || '';
            const title = book.title || 'Unknown Title';
            const author = book.author || book.authors || 'Unknown Author';
            const imageUrl = book.image_url || book.coverUrl || 'placeholder.svg';
            const rating = Number(book.average_rating || book.rating || 0).toFixed(1);
            
            return `
                <div class="book-card" onclick="navigateToBook('${bookId}')">
                    <div class="book-cover">
                        <img src="${imageUrl}" 
                             alt="${title}" 
                             onerror="this.src='placeholder.svg'">
                    </div>
                    <div class="book-info">
                        <h3 class="book-title">${title}</h3>
                        <p class="book-author">by ${author}</p>
                        <div class="book-rating">
                            <i class="fa-solid fa-star"></i>
                            <span>${rating}</span>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
        
        featuredBooksGrid.innerHTML = `
            <h1>Recommended For You</h1>
            <p class="subtitle">Based on your reading history and preferences</p>
            <div class="recommendations-grid">
                ${recommendedBooksHTML}
            </div>
        `;

    } catch (error) {
        console.error('Error in loadHomePage:', error);
        featuredBooksGrid.innerHTML = `
            <h1>Recommended For You</h1>
            <p class="subtitle">Based on your reading history and preferences</p>
            <div class="error-message">
                <i class="fa-solid fa-exclamation-circle"></i>
                <p>Failed to load recommendations. Please try again later.</p>
                <button class="btn btn-primary" onclick="navigateTo('explore')">
                    Explore Books Instead
                </button>
            </div>
        `;
    }
}

// Helper function to load popular books
async function loadPopularBooks(featuredBooksGrid, reason = "No personal recommendations available") {
    try {
        console.log(`Loading popular books as fallback. Reason: ${reason}`);
        
        // Show loading indicator
        featuredBooksGrid.innerHTML = `
            <h1>Popular Books</h1>
            <p class="subtitle">Discover reader favorites from across our community</p>
            <div class="loading">Loading books...</div>
        `;
        
        // Fetch popular books
        const response = await fetch(`${API_BASE_URL}/api/books?page=1&per_page=20`);
        if (!response.ok) {
            console.error(`Error fetching popular books: ${response.status}`);
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const popularBooksData = await response.json();
        console.log(`Received popular books data:`, popularBooksData);
        
        const books = popularBooksData.books || [];
        console.log(`Found ${books.length} popular books`);
        
        if (books.length > 0) {
            // Display popular books with different header
            const popularBooksHTML = books.map(book => {
                // Ensure we have values for all required properties
                const bookId = book.book_id || book.id || '';
                const title = book.title || 'Unknown Title';
                const author = book.author || book.authors || 'Unknown Author';
                const imageUrl = book.image_url || book.coverUrl || 'placeholder.svg';
                const rating = Number(book.average_rating || book.rating || 0).toFixed(1);
                
                return `
                    <div class="book-card" onclick="navigateToBook('${bookId}')">
                        <div class="book-cover">
                            <img src="${imageUrl}" 
                                alt="${title}" 
                                onerror="this.src='placeholder.svg'">
                        </div>
                        <div class="book-info">
                            <h3 class="book-title">${title}</h3>
                            <p class="book-author">by ${author}</p>
                            <div class="book-rating">
                                <i class="fa-solid fa-star"></i>
                                <span>${rating}</span>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
            
            featuredBooksGrid.innerHTML = `
                <h1>Popular Books</h1>
                <p class="subtitle">Discover reader favorites from across our community</p>
                <div class="recommendations-grid">
                    ${popularBooksHTML}
                </div>
            `;
            return;
        } else {
            // No books found at all
            featuredBooksGrid.innerHTML = `
                <h1>Popular Books</h1>
                <p class="subtitle">Discover reader favorites from across our community</p>
                <div class="no-books">
                    <i class="fa-solid fa-books"></i>
                    <p>We couldn't find any books to recommend. Try exploring some books to get started!</p>
                    <button class="btn btn-primary" onclick="navigateTo('explore')">
                        Explore Books
                    </button>
                </div>
            `;
        }
    } catch (error) {
        console.error('Error loading popular books:', error);
        featuredBooksGrid.innerHTML = `
            <h1>Popular Books</h1>
            <p class="subtitle">Discover reader favorites from across our community</p>
            <div class="error-message">
                <i class="fa-solid fa-exclamation-circle"></i>
                <p>Failed to load books. Please try again later.</p>
                <button class="btn btn-primary" onclick="navigateTo('explore')">
                    Explore Books Instead
                </button>
            </div>
        `;
    }
}

// Add these navigation functions
function navigateTo(pageId) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    
    // Show the selected page
    const selectedPage = document.getElementById(`${pageId}-page`);
    if (selectedPage) {
        selectedPage.classList.add('active');
        
        // Update the navigation link
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => link.classList.remove('active'));
        
        const activeLink = document.querySelector(`.nav-link[data-page="${pageId}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
        }
        
        // Load page-specific content if needed
        switch(pageId) {
            case 'home':
                loadHomePage();
                break;
            case 'explore':
                loadExplorePage();
                break;
            case 'search':
                loadSearchPage();
                break;
            case 'about':
                loadAboutPage();
                break;
            case 'profile':
                loadProfilePage();
                break;
        }
    }
    
    // Scroll to top
    window.scrollTo(0, 0);
}

function navigateToBook(bookId) {
    // Store current page state before navigating
    const currentPageId = document.querySelector('.page.active')?.id?.replace('-page', '');
    
    // Push state to history
    window.history.pushState(
        { page: currentPageId },
        '',
        `?book=${bookId}`
    );
    
    // Show book detail page
    const bookDetailPage = document.getElementById('book-detail-page');
    if (bookDetailPage) {
        document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
        bookDetailPage.classList.add('active');
        loadBookDetails(bookId);
    }
    
    // Scroll to top
    window.scrollTo(0, 0);
}

function navigateBack() {
    // Store the current active page before navigating
    const currentPage = document.querySelector('.page.active');
    
    // Try using browser's history first
    if (window.history.state && window.history.state.page) {
        // Hide current page
        if (currentPage) {
            currentPage.classList.remove('active');
        }
        
        // Show the previous page based on history state
        const previousPageId = window.history.state.page;
        const previousPage = document.getElementById(`${previousPageId}-page`);
        if (previousPage) {
            previousPage.classList.add('active');
            // Update navigation if needed
            const navLink = document.querySelector(`.nav-link[data-page="${previousPageId}"]`);
            if (navLink) {
                document.querySelectorAll('.nav-link').forEach(link => link.classList.remove('active'));
                navLink.classList.add('active');
            }
            window.history.back();
            return;
        }
    }
    
    // Fallback to default behavior if history state is not available
    if (currentPage) {
        currentPage.classList.remove('active');
    }
    
    // Default to home page if no previous page is found
    const homePage = document.getElementById('home-page');
    if (homePage) {
        homePage.classList.add('active');
        // Update navigation
        const homeLink = document.querySelector('.nav-link[data-page="home"]');
        if (homeLink) {
            document.querySelectorAll('.nav-link').forEach(link => link.classList.remove('active'));
            homeLink.classList.add('active');
        }
    }
    
    // Scroll to top
    window.scrollTo(0, 0);
}

async function loadProfilePage() {
    const userData = JSON.parse(localStorage.getItem('userData'));
    if (!userData) {
        window.location.href = 'login.html';
        return;
    }

    try {
        // Fetch user profile data
        const response = await fetch(`${API_BASE_URL}/api/profile?user_id=${userData.user_id}`);
        if (!response.ok) throw new Error('Failed to fetch profile');
        
        const profileData = await response.json();
        
        // Get the profile page element
        const profilePage = document.getElementById('profile-page');
        if (!profilePage) return;

        // Update the profile page with user details
        profilePage.innerHTML = `
            <div class="container">
                <div class="profile-header">
                    <h1><i class="fa-solid fa-user"></i> Profile</h1>
                </div>
                <div class="profile-content">
                    <div class="profile-info">
                        <div class="profile-detail">
                            <h2>Username</h2>
                            <p>${profileData.username}</p>
                        </div>
                        <div class="profile-detail">
                            <h2>User ID</h2>
                            <p>${profileData.user_id}</p>
                        </div>
                    </div>
                    <button class="btn btn-primary logout-btn" onclick="logout()">
                        <i class="fa-solid fa-sign-out-alt"></i> Logout
                    </button>
                </div>
            </div>
        `;

    } catch (error) {
        console.error('Error loading profile:', error);
        const profilePage = document.getElementById('profile-page');
        if (profilePage) {
            profilePage.innerHTML = `
                <div class="container">
                    <div class="error-message">
                        <i class="fa-solid fa-exclamation-circle"></i>
                        Failed to load profile. Please try again later.
                    </div>
                </div>
            `;
        }
    }
}

function logout() {
    localStorage.removeItem('isLoggedIn');
    localStorage.removeItem('userData');
    window.location.href = 'login.html';
}

// Show Toast Notification
function showToast(title, message) {
  toastTitle.textContent = title;
  toastDescription.textContent = message;
  toast.classList.add('show');
  
  // Auto hide after 3 seconds
  setTimeout(() => {
    hideToast();
  }, 3000);
}

// Hide Toast Notification
function hideToast() {
  toast.classList.remove('show');
}

// Utility: Debounce Function
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

// Create placeholder.svg for fallback images
const placeholderSvg = `
<svg xmlns="http://www.w3.org/2000/svg" width="200" height="300" viewBox="0 0 200 300">
  <rect width="200" height="300" fill="#f0e9e0" />
  <text x="100" y="150" font-family="Arial" font-size="14" text-anchor="middle" fill="#6d6259">No Image Available</text>
</svg>
`;

// Create a placeholder image element
const placeholderImg = document.createElement('img');
placeholderImg.src = `data:image/svg+xml;base64,${btoa(placeholderSvg)}`;
placeholderImg.id = 'placeholder-svg';
placeholderImg.style.display = 'none';
document.body.appendChild(placeholderImg);

// Add placeholdear.svg to the root for image fallbacks
const link = document.createElement('link');
link.rel = 'preload';
link.href = `data:image/svg+xml;base64,${btoa(placeholderSvg)}`;
link.as = 'image';
document.head.appendChild(link);

// Add event listener for search input
document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('search-input');
    if (searchInput) {
        let debounceTimer;
        searchInput.addEventListener('input', () => {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(handleSearch, 300);
        });
    }
});

async function handleSearch() {
    const searchInput = document.getElementById('search-input');
    const searchResults = document.getElementById('search-results');
    
    if (!searchInput || !searchResults) return;
    
    const query = searchInput.value.trim();
    if (!query) {
        searchResults.innerHTML = '';
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/api/search?q=${encodeURIComponent(query)}`);
        if (!response.ok) throw new Error('Search failed');
        
        const books = await response.json();
        
        if (!Array.isArray(books) || books.length === 0) {
            searchResults.innerHTML = '<div class="no-results">No books found</div>';
            return;
        }

        searchResults.innerHTML = books.map(book => `
            <div class="book-card" data-book-id="${book.book_id || ''}" onclick="navigateToBook('${book.book_id}')">
                <div class="book-cover">
                    <img src="${book.image_url || 'placeholder.svg'}" 
                         alt="${book.title || 'Book cover'}" 
                         onerror="this.src='placeholder.svg'">
                </div>
                <div class="book-info">
                    <h3 class="book-title">${book.title || 'Untitled'}</h3>
                    <p class="book-author">by ${book.author || 'Unknown'}</p>
                    <div class="book-rating">
                        <i class="fa-solid fa-star"></i>
                        <span>${Number(book.average_rating || 0).toFixed(1)}</span>
                    </div>
                    ${Array.isArray(book.genres) && book.genres.length > 0 ? `
                        <div class="book-genres">
                            ${book.genres.slice(0, 2).map(genre => 
                                `<span class="book-genre">${genre}</span>`
                            ).join('')}
                        </div>
                    ` : ''}
                </div>
            </div>
        `).join('');

    } catch (error) {
        console.error('Search error:', error);
        searchResults.innerHTML = '<div class="error">Error searching books</div>';
    }
}

let currentPage = 1;
const perPage = 20;

// Update the book card click handler in loadBooks function
async function loadBooks(page = 1, genre = '') {
    const booksGrid = document.getElementById('explore-books-grid');
    const paginationDiv = document.getElementById('pagination');
    
    if (!booksGrid || !paginationDiv) return;

    try {
        booksGrid.innerHTML = '<div class="loading">Loading books...</div>';
        
        // Add genre to query parameters if selected
        const url = new URL(`${API_BASE_URL}/api/books`);
        url.searchParams.set('page', page);
        url.searchParams.set('per_page', 20);
        if (genre) {
            url.searchParams.set('genre', genre);
        }
        
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const data = await response.json();

        if (!data.books || data.books.length === 0) {
            booksGrid.innerHTML = '<div class="no-books">No books found</div>';
            paginationDiv.innerHTML = '';
            return;
        }

        // Clear grid before adding new books
        booksGrid.innerHTML = '';
        
        // Create a properly sized grid for the books
        const bookGrid = document.createElement('div');
        bookGrid.className = 'books-grid';
        
        // Add each book card to the grid using the helper function
        data.books.forEach(book => {
            const bookCard = createBookCard(book);
            bookGrid.appendChild(bookCard);
        });
        
        // Add the grid to the container
        booksGrid.appendChild(bookGrid);

        // Render pagination
        renderPagination(data.pagination);

    } catch (error) {
        console.error('Error loading books:', error);
        booksGrid.innerHTML = '<div class="error">Failed to load books</div>';
        paginationDiv.innerHTML = '';
    }
}

async function loadBookDetails(bookId) {
    try {
        const bookDetailPage = document.getElementById('book-detail-page');
        const container = bookDetailPage.querySelector('.book-detail-container');
        
        if (!container) {
            console.error('Book detail container not found');
            return;
        }
        
        container.innerHTML = '<div class="loading">Loading book details...</div>';
        showPage('book-detail-page');

        const response = await fetch(`${API_BASE_URL}/api/books/${bookId}`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const book = await response.json();
        
        container.innerHTML = `
            <button class="back-button" onclick="navigateBack()">
                <i class="fas fa-arrow-left"></i> Back
            </button>
            <div class="book-detail-content">
                <div class="book-detail-left">
                    <div class="book-cover-large">
                        <img src="${book.image_url || 'placeholder.svg'}" 
                             alt="${book.title}"
                             onerror="this.src='placeholder.svg'">
                    </div>
                    <div class="book-rating-large">
                        <i class="fas fa-star"></i>
                        <span>${Number(book.average_rating || 0).toFixed(1)}</span>
                    </div>
                    <div class="genres-section">
                        <h3>Genres</h3>
                        <div class="book-genres-large">
                            ${book.genres && book.genres.length > 0 
                                ? book.genres.map(genre => `<span class="book-genre">${genre}</span>`).join('')
                                : '<div class="no-genres">No genres available</div>'
                            }
                        </div>
                    </div>
                </div>
                <div class="book-detail-right">
                    <h1>${book.title}</h1>
                    <h2>by ${book.author}</h2>
                    <div class="book-description">
                        <h3>Description</h3>
                        <p>${book.description || 'No description available.'}</p>
                    </div>
                    <div class="you-may-like">
                        <h3>You May Also Like</h3>
                        <div class="loading">Loading recommendations...</div>
                    </div>
                </div>
            </div>
        `;
        
        // After loading book details, fetch recommendations
        await displayBookRecommendations(bookId);
        
    } catch (error) {
        console.error('Error loading book details:', error);
        const container = document.querySelector('.book-detail-container');
        if (container) {
            container.innerHTML = `
                <button class="back-button" onclick="navigateBack()">
                    <i class="fas fa-arrow-left"></i> Back
                </button>
                <div class="error-message">
                    <i class="fas fa-exclamation-circle"></i>
                    Failed to load book details. Please try again later.
                </div>
            `;
        }
    }
}

// Update showPage function to handle book detail page
function showPage(pageId) {
    document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
    document.querySelectorAll('.nav-link').forEach(link => link.classList.remove('active'));

    const targetPage = document.getElementById(pageId);
    if (targetPage) {
        targetPage.classList.add('active');
        currentActivePage = pageId;

        // Update active nav link except for book detail page
        if (pageId !== 'book-detail-page') {
            const activeLink = document.querySelector(`.nav-link[data-page="${pageId}"]`);
            if (activeLink) activeLink.classList.add('active');
        }
    }
}

function renderPagination(pagination) {
    const paginationDiv = document.getElementById('pagination');
    if (!paginationDiv) return;

    const { current_page, total_pages } = pagination;
    let html = '<div class="pagination">';

    // Previous button
    html += `
        <button class="page-btn prev-btn" 
                ${current_page === 1 ? 'disabled' : ''}
                onclick="loadBooks(${current_page - 1})">
            Previous
        </button>
    `;

    // Page numbers with ellipsis
    for (let i = 1; i <= total_pages; i++) {
        if (
            i === 1 || // First page
            i === total_pages || // Last page
            (i >= current_page - 2 && i <= current_page + 2) // Pages around current
        ) {
            html += `
                <button class="page-btn ${i === current_page ? 'active' : ''}"
                        onclick="loadBooks(${i})">
                    ${i}
                </button>
            `;
        } else if (
            i === current_page - 3 ||
            i === current_page + 3
        ) {
            html += '<span class="page-dots">...</span>';
        }
    }

    // Next button
    html += `
        <button class="page-btn next-btn" 
                ${current_page === total_pages ? 'disabled' : ''}
                onclick="loadBooks(${current_page + 1})">
            Next
        </button>
    </div>`;

    paginationDiv.innerHTML = html;
}

// Remove any existing scroll event listeners
window.removeEventListener('scroll', loadBooks);

// Add this utility function at the top of your file
function getRequiredElements(elementIds) {
    const elements = {};
    const missingElements = [];

    elementIds.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            elements[id] = element;
        } else {
            missingElements.push(id);
        }
    });

    if (missingElements.length > 0) {
        console.error('Missing required elements:', missingElements);
        return null;
    }

    return elements;
}

async function displayBookRecommendations(bookId) {
    try {
        console.log(`Loading recommendations for book ID: ${bookId}`);
        
        const response = await fetch(`${API_BASE_URL}/api/books/${bookId}/recommendations`);
        if (!response.ok) throw new Error('Failed to fetch recommendations');
        
        const data = await response.json();
        console.log('Book recommendations API response:', data);
        
        // Get recommendations from the API response structure
        const recommendations = data.recommendations || [];
        
        // Get the book detail container
        const bookDetailContainer = document.querySelector('.book-detail-right');
        if (!bookDetailContainer) {
            console.error('Book detail container not found');
            return;
        }
        
        // Find or create the recommendations section
        let recommendationsSection = bookDetailContainer.querySelector('.you-may-like');
        if (!recommendationsSection) {
            recommendationsSection = document.createElement('div');
            recommendationsSection.className = 'you-may-like';
            bookDetailContainer.appendChild(recommendationsSection);
        }
        
        // Clear previous content
        recommendationsSection.innerHTML = '<h3>You May Also Like</h3>';
        
        if (recommendations && recommendations.length > 0) {
            console.log(`Displaying ${recommendations.length} recommendations`);
            
            // Create the recommendations grid
            const recommendationsGrid = document.createElement('div');
            recommendationsGrid.className = 'recommendations-grid';
            
            // Add each book card to the grid
            recommendations.forEach((book, index) => {
                console.log(`Creating card for book ${index + 1}:`, book.title);
                const bookCard = createBookCard(book);
                recommendationsGrid.appendChild(bookCard);
            });
            
            // Add the grid to the recommendations section
            recommendationsSection.appendChild(recommendationsGrid);
        } else {
            console.log('No recommendations available');
            // No recommendations available
            const noRecsMessage = document.createElement('p');
            noRecsMessage.textContent = 'No recommendations available for this book.';
            recommendationsSection.appendChild(noRecsMessage);
        }
        
    } catch (error) {
        console.error('Error loading recommendations:', error);
        
        // Display error message in recommendations section
        const bookDetailContainer = document.querySelector('.book-detail-right');
        if (bookDetailContainer) {
            let recommendationsSection = bookDetailContainer.querySelector('.you-may-like');
            if (recommendationsSection) {
                recommendationsSection.innerHTML = `
                    <h3>You May Also Like</h3>
                    <p class="error">Failed to load recommendations.</p>
                `;
            }
        }
    }
}

// Function to load similar books using the same endpoint
function loadSimilarBooks(bookId) {
    if (!bookId) {
        console.error('No book ID provided for recommendations');
        return;
    }
    
    console.log(`Loading similar books for book ID: ${bookId}`);
    
    // Find the recommendations container
    const recommendationsContainer = document.querySelector('.you-may-like .recommendations-grid');
    if (!recommendationsContainer) {
        console.error('Recommendations container not found');
        return;
    }
    
    // Show loading state
    recommendationsContainer.innerHTML = '<div class="loading">Finding similar books...</div>';
    
    // Fetch recommendations from API
    fetch(`${API_BASE_URL}/api/books/${bookId}/recommendations?limit=5`)
        .then(response => {
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return response.json();
        })
        .then(data => {
            console.log('Similar books API response:', data);
            
            // Handle API response structure correctly
            const recommendations = data.recommendations || [];
            
            if (!recommendations || recommendations.length === 0) {
                recommendationsContainer.innerHTML = '<p class="no-books">No similar books found.</p>';
                return;
            }
            
            // Clear loading state
            recommendationsContainer.innerHTML = '';
            
            // Add each recommendation to the grid using the helper function
            recommendations.forEach(book => {
                const bookCard = createBookCard(book);
                recommendationsContainer.appendChild(bookCard);
            });
        })
        .catch(error => {
            console.error('Error loading similar books:', error);
            recommendationsContainer.innerHTML = '<p class="error">Failed to load similar books.</p>';
        });
}

// Add a helper function to create consistent book cards
function createBookCard(book) {
    const bookCard = document.createElement('div');
    bookCard.className = 'book-card';
    
    // Handle different API response formats
    const bookId = book.book_id || book.id;
    const title = book.title || 'Unknown Title';
    const author = book.author || 'Unknown Author';
    const imageUrl = book.image_url || book.coverUrl || 'placeholder.svg';
    const rating = Number(book.average_rating || book.rating || 0).toFixed(1);
    
    bookCard.onclick = () => navigateToBook(bookId);
    
    bookCard.innerHTML = `
        <div class="book-cover">
            <img src="${imageUrl}" 
                alt="${title}"
                onerror="this.src='placeholder.svg'">
        </div>
        <div class="book-info">
            <h3 class="book-title">${title}</h3>
            <p class="book-author">by ${author}</p>
            <div class="book-rating">
                <i class="fas fa-star"></i> ${rating}
            </div>
        </div>
    `;
    
    return bookCard;
}

// Find the displayBookDetails function and modify it to include the call to loadSimilarBooks
function displayBookDetails(bookData) {
    // Set book details page as active
    setActivePage('book-detail');
    
    // Set book details
    document.getElementById('detail-book-title').textContent = bookData.title || 'Unknown Title';
    document.getElementById('detail-book-author').textContent = bookData.author || 'Unknown Author';
    document.getElementById('detail-book-rating').textContent = bookData.average_rating?.toFixed(1) || '0.0';
    document.getElementById('detail-book-description').textContent = bookData.description || 'No description available';
    
    // Set book cover image
    const coverImg = document.getElementById('detail-book-cover');
    if (bookData.image_url) {
        coverImg.src = bookData.image_url;
        coverImg.alt = bookData.title;
    } else {
        coverImg.src = 'img/book-cover-placeholder.jpg';
        coverImg.alt = 'No cover available';
    }
    
    // Display genres if available
    const genresContainer = document.getElementById('detail-book-genres');
    genresContainer.innerHTML = '';
    
    if (bookData.genres && bookData.genres.length > 0) {
        bookData.genres.forEach(genre => {
            const genreSpan = document.createElement('span');
            genreSpan.className = 'book-genre';
            genreSpan.textContent = genre;
            genresContainer.appendChild(genreSpan);
        });
    } else {
        const genreSpan = document.createElement('span');
        genreSpan.className = 'book-genre';
        genreSpan.textContent = 'Fiction';
        genresContainer.appendChild(genreSpan);
    }
    
    // Add section for similar books if it doesn't exist
    if (!document.querySelector('.you-may-like')) {
        const bookDetailContainer = document.querySelector('.book-detail-container');
        
        // Create the "You May Also Like" section
        const similarBooksSection = document.createElement('div');
        similarBooksSection.className = 'you-may-like';
        similarBooksSection.innerHTML = `
            <h3>You May Also Like</h3>
            <div class="recommendations-grid"></div>
        `;
        
        // Append it to the book detail container
        bookDetailContainer.appendChild(similarBooksSection);
    }
    
    // Load similar books for this book
    loadSimilarBooks(bookData.book_id);
}