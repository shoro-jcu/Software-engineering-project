# Smart Library - Book Recommendation Engine

A data-driven book recommendation system that uses collaborative filtering to suggest books to users based on their borrowing history and ratings.

## Features

### User Features
- **User Registration & Authentication** - Secure login and registration system
- **Book Browsing** - Search and filter books by title, author, and genre
- **Book Details** - View detailed information about each book with cover images
- **Rating System** - Rate books 1-5 stars with optional reviews
- **Personalized Recommendations** - Get book recommendations based on collaborative filtering
- **My Ratings** - View all your past ratings and reading statistics
- **Profile Management** - Update your profile information

### Admin Features
- **Admin Dashboard** - Overview of library statistics and key metrics
- **Manage Books** - Add, edit, and delete books from the catalog
- **Manage Users** - View and manage user accounts, toggle admin privileges
- **Analytics Dashboard** - Visual charts showing ratings distribution and popular books
- **Recommendation Engine** - Configure algorithm settings and precompute recommendations

## Technology Stack

- **Framework**: Flask 2.x
- **Database**: SQLite 3
- **Frontend**: Bootstrap 5, Bootstrap Icons
- **Charting**: Chart.js
- **Recommendation Algorithm**: Collaborative Filtering (Cosine Similarity, Pearson Correlation)
- **Language**: Python 3.9+

## Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd library_app
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   
   On Linux/Mac:
   ```bash
   source venv/bin/activate
   ```
   
   On Windows:
   ```cmd
   venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install flask numpy
   ```

5. **Initialize the database**
   ```bash
   python -c "from app import init_db; init_db()"
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

7. **Access the application**
   Open your browser and navigate to `http://localhost:5001`

## Default Credentials

### Admin Account
- **Username**: admin
- **Password**: admin123

### Test Users
- **Username**: user1
- **Password**: user123

## Project Structure

```
library_app/
├── app.py                 # Main application file
├── schema.sql             # Database schema
├── library.db             # SQLite database (generated)
├── templates/             # HTML templates
│   ├── base.html          # Base template with CSS
│   ├── index.html         # Homepage
│   ├── books.html         # Book browsing page
│   ├── book_detail.html   # Book detail page
│   ├── dashboard.html     # User dashboard
│   ├── profile.html       # User profile
│   ├── my_ratings.html    # User ratings history
│   ├── login.html         # Login page
│   ├── register.html      # Registration page
│   └── admin/             # Admin templates
│       ├── dashboard.html
│       ├── books.html
│       ├── users.html
│       ├── analytics.html
│       └── recommendations.html
└── venv/                  # Virtual environment (generated)
```

## Recommendation Algorithm

The system uses collaborative filtering with two similarity metrics:

1. **Cosine Similarity** - Measures the angle between user rating vectors
2. **Pearson Correlation** - Measures linear correlation between users

### Algorithm Flow

1. **Data Collection** - Gather user ratings from the database
2. **Matrix Construction** - Build user-item rating matrix
3. **Similarity Calculation** - Compute similarity between users
4. **Neighborhood Selection** - Select most similar users (k-nearest neighbors)
5. **Prediction Generation** - Predict ratings for unrated items
6. **Recommendation Ranking** - Return top-n recommendations

### Cold Start Handling

- New users receive popular books based on rating counts
- Genre preferences are used when no rating history exists
- Precomputed recommendations for faster response times

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/recommendations` | GET | Get personalized book recommendations |
| `/api/book/<id>/ratings` | GET | Get all ratings for a specific book |
| `/api/ratings` | GET | Get all ratings (admin only) |

## Usage

### Browsing Books
1. Click on "Books" in the navigation menu
2. Use search to find books by title or author
3. Filter books by genre using the dropdown
4. Click on any book to view details

### Rating Books
1. Go to a book's detail page
2. Click on the stars to rate the book (1-5)
3. Optionally add a review comment
4. Submit the rating

### Getting Recommendations
1. Log in to your account
2. Go to "Dashboard"
3. View personalized recommendations under "Recommended for You"
4. Rate more books to improve recommendation accuracy

### Admin Management
1. Log in with admin credentials
2. Click on "Dashboard" to access admin panel
3. Manage books, users, and view analytics

## Configuration

You can configure the recommendation engine in `app.py`:

- `MIN_RATINGS_FOR_RECOMMENDATION` - Minimum ratings needed for personalized recommendations
- `RECOMMENDATION_COUNT` - Number of recommendations to show
- `SIMILAR_USER_COUNT` - Number of similar users to consider

## License

This project is for educational purposes.

## Contributing

Feel free to submit issues and pull requests to improve the system.

## Contact

For questions or support, please contact the project maintainer.
