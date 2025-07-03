function checkAuth() {
    const isLoggedIn = localStorage.getItem('isLoggedIn') === 'true';
    return isLoggedIn;
}

function handleProfileClick(event) {
    event.preventDefault();
    
    if (!checkAuth()) {
        window.location.href = 'login.html';
    } else {
        // Show profile page
        document.querySelectorAll('.page').forEach(page => {
            page.classList.remove('active');
        });
        document.getElementById('profile-page').classList.add('active');
    }
}

// Common utility functions
function showError(elementId, message) {
    const errorElement = document.getElementById(elementId);
    if (errorElement) {
        errorElement.textContent = message;
        errorElement.style.display = 'block';
    }
}

// Handle login functionality
document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.getElementById('login-form');
    
    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const username = document.getElementById('login-username').value;
            const password = document.getElementById('login-password').value;

            try {
                const response = await fetch(`${API_BASE_URL}/api/login`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ username, password })
                });

                const data = await response.json();

                if (response.ok) {
                    localStorage.setItem('userData', JSON.stringify(data.user));
                    localStorage.setItem('isLoggedIn', 'true');
                    window.location.href = 'home.html';
                } else {
                    showError('login-error', data.error || 'Login failed');
                }
            } catch (error) {
                console.error('Login error:', error);
                showError('login-error', 'Login failed. Please try again.');
            }
        });
    }

    // Handle signup functionality
    const signupForm = document.getElementById('signup-form');
    
    if (signupForm) {
        signupForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const username = document.getElementById('signup-username').value;
            const email = document.getElementById('signup-email').value;
            const password = document.getElementById('signup-password').value;

            try {
                const response = await fetch(`${API_BASE_URL}/api/signup`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ username, email, password })
                });

                const data = await response.json();

                if (response.ok) {
                    localStorage.setItem('userData', JSON.stringify(data.user));
                    localStorage.setItem('isLoggedIn', 'true');
                    window.location.href = 'home.html';
                } else {
                    showError('signup-error', data.error || 'Registration failed');
                }
            } catch (error) {
                console.error('Signup error:', error);
                showError('signup-error', 'Registration failed. Please try again.');
            }
        });
    }
});