import os
import sqlite3
import random
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session, g, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

app = Flask(__name__)
app.secret_key = 'library_recommendation_secret_key_2024'
app.config['DATABASE'] = os.path.join(os.path.dirname(__file__), 'library.db')

# Database helpers
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(app.config['DATABASE'])
        db.row_factory = sqlite3.Row
    return db

def query_db(query, args=(), one=False):
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv

def execute_db(query, args=()):
    db = get_db()
    cur = db.execute(query, args)
    db.commit()
    cur.close()
    return cur.lastrowid

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# Init database
def init_db():
    with app.app_context():
        db = get_db()
        with app.open_resource('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please sign in first', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please sign in first', 'warning')
            return redirect(url_for('login'))
        user = query_db('SELECT * FROM users WHERE id = ?', [session['user_id']], one=True)
        if not user or not user['is_admin']:
            flash('Admin access required', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# Recommendation Engine
class RecommendationEngine:
    @staticmethod
    def get_user_ratings_matrix():
        """Get user-book ratings matrix"""
        ratings = query_db('''
            SELECT user_id, book_id, rating FROM ratings
        ''')
        if not ratings:
            return {}, {}, {}
        
        # Build matrix
        user_ratings = {}
        book_ids = set()
        for r in ratings:
            uid = r['user_id']
            bid = r['book_id']
            if uid not in user_ratings:
                user_ratings[uid] = {}
            user_ratings[uid][bid] = r['rating']
            book_ids.add(bid)
        
        return user_ratings, list(book_ids), ratings
    
    @staticmethod
    def get_user_genre_preferences(user_id):
        """Get user's genre preferences based on their ratings"""
        ratings = query_db('''
            SELECT r.rating, b.genre
            FROM ratings r
            JOIN books b ON r.book_id = b.id
            WHERE r.user_id = ?
        ''', [user_id])
        
        if not ratings:
            return {}
        
        genre_scores = {}
        for r in ratings:
            genre = r['genre']
            if genre not in genre_scores:
                genre_scores[genre] = {'total': 0, 'count': 0}
            genre_scores[genre]['total'] += r['rating']
            genre_scores[genre]['count'] += 1
        
        # Calculate average rating per genre
        preferences = {}
        for genre, data in genre_scores.items():
            preferences[genre] = data['total'] / data['count']
        
        return preferences
    
    @staticmethod
    def cosine_similarity(user1_ratings, user2_ratings):
        """Compute cosine similarity between two users"""
        common_books = set(user1_ratings.keys()) & set(user2_ratings.keys())
        if len(common_books) < 2:
            return 0.0
        
        vec1 = np.array([user1_ratings[b] for b in common_books])
        vec2 = np.array([user2_ratings[b] for b in common_books])
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    @staticmethod
    def pearson_correlation(user1_ratings, user2_ratings):
        """Compute Pearson correlation between two users"""
        common_books = set(user1_ratings.keys()) & set(user2_ratings.keys())
        if len(common_books) < 2:
            return 0.0
        
        vec1 = np.array([user1_ratings[b] for b in common_books])
        vec2 = np.array([user2_ratings[b] for b in common_books])
        
        mean1 = np.mean(vec1)
        mean2 = np.mean(vec2)
        
        centered1 = vec1 - mean1
        centered2 = vec2 - mean2
        
        num = np.dot(centered1, centered2)
        den = np.linalg.norm(centered1) * np.linalg.norm(centered2)
        
        if den == 0:
            return 0.0
        
        return float(num / den)
    
    @classmethod
    def compute_similarity(cls, user1_ratings, user2_ratings, method='cosine'):
        """Compute similarity using specified method"""
        if method == 'pearson':
            return cls.pearson_correlation(user1_ratings, user2_ratings)
        return cls.cosine_similarity(user1_ratings, user2_ratings)
    
    @classmethod
    def get_similar_users(cls, user_id, n=5, method='cosine', min_similarity=0.1):
        """Find n most similar users to given user"""
        user_ratings, _, _ = cls.get_user_ratings_matrix()
        
        if user_id not in user_ratings or len(user_ratings) < 2:
            return []
        
        target_ratings = user_ratings[user_id]
        similarities = []
        
        for other_id, other_ratings in user_ratings.items():
            if other_id == user_id:
                continue
            sim = cls.compute_similarity(target_ratings, other_ratings, method)
            if sim > min_similarity:
                similarities.append((other_id, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]
    
    @classmethod
    def recommend_books(cls, user_id, n=5, method='cosine'):
        """Recommend top-N books for user using collaborative filtering"""
        user_ratings, all_book_ids, _ = cls.get_user_ratings_matrix()
        
        # Cold start: if user has no ratings or very few users
        if user_id not in user_ratings or len(user_ratings) < 3:
            return cls.popularity_based_recommendation(user_id, n)
        
        similar_users = cls.get_similar_users(user_id, n=10, method=method)
        if not similar_users:
            return cls.popularity_based_recommendation(user_id, n)
        
        target_ratings = user_ratings[user_id]
        recommendations = {}
        
        for other_id, similarity in similar_users:
            other_ratings = user_ratings[other_id]
            for book_id, rating in other_ratings.items():
                if book_id in target_ratings:
                    continue
                if book_id not in recommendations:
                    recommendations[book_id] = {'score': 0, 'total_sim': 0, 'reasons': []}
                recommendations[book_id]['score'] += rating * similarity
                recommendations[book_id]['total_sim'] += similarity
                recommendations[book_id]['reasons'].append((other_id, similarity, rating))
        
        if not recommendations:
            return cls.popularity_based_recommendation(user_id, n)
        
        # Normalize scores
        scored_books = []
        for book_id, data in recommendations.items():
            if data['total_sim'] > 0:
                normalized_score = data['score'] / data['total_sim']
                # Find top reason
                top_reason = max(data['reasons'], key=lambda x: x[1])
                scored_books.append({
                    'book_id': book_id,
                    'predicted_rating': round(normalized_score, 2),
                    'similarity': round(top_reason[1], 3),
                    'reason_user_id': top_reason[0]
                })
        
        scored_books.sort(key=lambda x: x['predicted_rating'], reverse=True)
        
        # Get book details
        result = []
        for item in scored_books[:n]:
            book = query_db('SELECT * FROM books WHERE id = ?', [item['book_id']], one=True)
            if book:
                # Find a book the target user liked that the similar user also rated
                reason_user_ratings = user_ratings.get(item['reason_user_id'], {})
                common_books = set(target_ratings.keys()) & set(reason_user_ratings.keys())
                reason_book = None
                if common_books:
                    best_common = max(common_books, key=lambda b: target_ratings[b])
                    reason_book = query_db('SELECT title FROM books WHERE id = ?', [best_common], one=True)
                
                result.append({
                    'book': dict(book),
                    'predicted_rating': item['predicted_rating'],
                    'similarity': item['similarity'],
                    'reason': f"Because you liked \"{reason_book['title']}\"" if reason_book else "Based on similar users' preferences"
                })
        
        return result
    
    @classmethod
    def popularity_based_recommendation(cls, user_id, n=5):
        """Popularity-based recommendation for cold start with genre preference boost"""
        # Get user's genre preferences
        genre_prefs = cls.get_user_genre_preferences(user_id)
        
        # Get books the user hasn't rated
        rated_books = query_db('SELECT book_id FROM ratings WHERE user_id = ?', [user_id])
        rated_ids = [r['book_id'] for r in rated_books] if rated_books else []
        
        if rated_ids:
            placeholders = ','.join('?' * len(rated_ids))
            books = query_db(f'''
                SELECT b.*, AVG(r.rating) as avg_rating, COUNT(r.id) as rating_count
                FROM books b
                LEFT JOIN ratings r ON b.id = r.book_id
                WHERE b.id NOT IN ({placeholders})
                GROUP BY b.id
                ORDER BY avg_rating DESC, rating_count DESC
                LIMIT ?
            ''', rated_ids + [n * 2])  # Get more books to filter by genre
        else:
            books = query_db('''
                SELECT b.*, AVG(r.rating) as avg_rating, COUNT(r.id) as rating_count
                FROM books b
                LEFT JOIN ratings r ON b.id = r.book_id
                GROUP BY b.id
                ORDER BY avg_rating DESC, rating_count DESC
                LIMIT ?
            ''', [n * 2])
        
        # Boost books in preferred genres
        scored_books = []
        for b in books:
            score = (b['avg_rating'] or 0) * 0.7 + (b['rating_count'] or 0) * 0.3
            if genre_prefs and b['genre'] in genre_prefs:
                score += genre_prefs[b['genre']] * 0.5  # Boost by genre preference
            scored_books.append({
                'book': dict(b),
                'score': score,
                'predicted_rating': round(b['avg_rating'] or 0, 2),
                'similarity': 1.0,
                'reason': 'Trending recommendation' if not genre_prefs else f"Popular in {b['genre']}"
            })
        
        # Sort by score and return top n
        scored_books.sort(key=lambda x: x['score'], reverse=True)
        return scored_books[:n]

# Routes
@app.route('/')
def index():
    books = query_db('''
        SELECT b.*, AVG(r.rating) as avg_rating, COUNT(r.id) as rating_count
        FROM books b
        LEFT JOIN ratings r ON b.id = r.book_id
        GROUP BY b.id
        ORDER BY avg_rating DESC
        LIMIT 8
    ''')
    return render_template('index.html', books=books)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if query_db('SELECT id FROM users WHERE username = ?', [username], one=True):
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        
        if query_db('SELECT id FROM users WHERE email = ?', [email], one=True):
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        user_id = execute_db(
            'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
            [username, email, hashed_password]
        )
        flash('Registration successful! Please sign in', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = query_db('SELECT * FROM users WHERE username = ?', [username], one=True)
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['is_admin'] = bool(user['is_admin'])
            flash(f'Welcome back, {user["username"]}!', 'success')
            if user['is_admin']:
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('dashboard'))
        
        flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been signed out', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    user = query_db('SELECT * FROM users WHERE id = ?', [session['user_id']], one=True)
    
    # Get user's ratings
    my_ratings = query_db('''
        SELECT r.*, b.title, b.author, b.cover_image
        FROM ratings r
        JOIN books b ON r.book_id = b.id
        WHERE r.user_id = ?
        ORDER BY r.created_at DESC
        LIMIT 5
    ''', [session['user_id']])
    
    # Get recommendations using session method preference
    method = session.get('rec_method', 'cosine')
    recommendations = RecommendationEngine.recommend_books(session['user_id'], n=6, method=method)
    
    return render_template('dashboard.html', user=user, my_ratings=my_ratings, recommendations=recommendations)

@app.route('/books')
def books():
    page = request.args.get('page', 1, type=int)
    per_page = 12
    offset = (page - 1) * per_page
    
    genre = request.args.get('genre', '')
    search = request.args.get('search', '')
    
    query = '''
        SELECT b.*, AVG(r.rating) as avg_rating, COUNT(r.id) as rating_count
        FROM books b
        LEFT JOIN ratings r ON b.id = r.book_id
        WHERE 1=1
    '''
    args = []
    
    if genre:
        query += ' AND b.genre = ?'
        args.append(genre)
    
    if search:
        query += ' AND (b.title LIKE ? OR b.author LIKE ?)'
        args.extend([f'%{search}%', f'%{search}%'])
    
    query += ' GROUP BY b.id ORDER BY avg_rating DESC LIMIT ? OFFSET ?'
    args.extend([per_page, offset])
    
    books = query_db(query, args)
    
    # Get genres for filter
    genres = query_db('SELECT DISTINCT genre FROM books ORDER BY genre')
    
    return render_template('books.html', books=books, genres=genres, current_genre=genre, search=search, page=page)

@app.route('/book/<int:book_id>')
def book_detail(book_id):
    book = query_db('SELECT * FROM books WHERE id = ?', [book_id], one=True)
    if not book:
        flash('Book not found', 'danger')
        return redirect(url_for('books'))
    
    ratings = query_db('''
        SELECT r.*, u.username
        FROM ratings r
        JOIN users u ON r.user_id = u.id
        WHERE r.book_id = ?
        ORDER BY r.created_at DESC
    ''', [book_id])
    
    avg_rating = query_db('''
        SELECT AVG(rating) as avg, COUNT(*) as count
        FROM ratings
        WHERE book_id = ?
    ''', [book_id], one=True)
    
    user_rating = None
    if 'user_id' in session:
        user_rating = query_db(
            'SELECT * FROM ratings WHERE user_id = ? AND book_id = ?',
            [session['user_id'], book_id], one=True
        )
    
    return render_template('book_detail.html', book=book, ratings=ratings, 
                          avg_rating=avg_rating, user_rating=user_rating)

@app.route('/rate_book/<int:book_id>', methods=['POST'])
@login_required
def rate_book(book_id):
    rating = int(request.form['rating'])
    review = request.form.get('review', '')
    
    if rating < 1 or rating > 5:
        flash('Rating must be between 1 and 5', 'danger')
        return redirect(url_for('book_detail', book_id=book_id))
    
    existing = query_db(
        'SELECT id FROM ratings WHERE user_id = ? AND book_id = ?',
        [session['user_id'], book_id], one=True
    )
    
    if existing:
        execute_db('''
            UPDATE ratings SET rating = ?, review = ?, updated_at = ?
            WHERE user_id = ? AND book_id = ?
        ''', [rating, review, datetime.now(), session['user_id'], book_id])
        flash('Rating updated', 'success')
    else:
        execute_db('''
            INSERT INTO ratings (user_id, book_id, rating, review, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', [session['user_id'], book_id, rating, review, datetime.now(), datetime.now()])
        flash('Rating submitted', 'success')
    
    return redirect(url_for('book_detail', book_id=book_id))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        email = request.form['email']
        bio = request.form.get('bio', '')
        
        execute_db('UPDATE users SET email = ?, bio = ? WHERE id = ?',
                  [email, bio, session['user_id']])
        flash('Profile updated', 'success')
        return redirect(url_for('profile'))
    
    user = query_db('SELECT * FROM users WHERE id = ?', [session['user_id']], one=True)
    
    # Get user's rating stats
    stats = query_db('''
        SELECT COUNT(*) as total_ratings, AVG(rating) as avg_rating
        FROM ratings
        WHERE user_id = ?
    ''', [session['user_id']], one=True)
    
    return render_template('profile.html', user=user, stats=stats)

@app.route('/my_ratings')
@login_required
def my_ratings():
    user = query_db('SELECT * FROM users WHERE id = ?', [session['user_id']], one=True)
    
    # Get all user's ratings
    ratings = query_db('''
        SELECT r.*, b.title, b.author, b.cover_image, b.genre
        FROM ratings r
        JOIN books b ON r.book_id = b.id
        WHERE r.user_id = ?
        ORDER BY r.created_at DESC
    ''', [session['user_id']])
    
    # Get stats
    stats = query_db('''
        SELECT COUNT(*) as total_ratings, AVG(rating) as avg_rating,
               MIN(rating) as min_rating, MAX(rating) as max_rating
        FROM ratings
        WHERE user_id = ?
    ''', [session['user_id']], one=True)
    
    # Get genre distribution
    genre_stats = query_db('''
        SELECT b.genre, COUNT(*) as count, AVG(r.rating) as avg_rating
        FROM ratings r
        JOIN books b ON r.book_id = b.id
        WHERE r.user_id = ?
        GROUP BY b.genre
        ORDER BY count DESC
    ''', [session['user_id']])
    
    return render_template('my_ratings.html', 
                          user=user, 
                          ratings=ratings, 
                          stats=stats,
                          genre_stats=genre_stats)

# Admin routes
@app.route('/admin')
@admin_required
def admin_dashboard():
    stats = {
        'total_users': query_db('SELECT COUNT(*) as c FROM users', one=True)['c'],
        'total_books': query_db('SELECT COUNT(*) as c FROM books', one=True)['c'],
        'total_ratings': query_db('SELECT COUNT(*) as c FROM ratings', one=True)['c'],
        'avg_rating': query_db('SELECT AVG(rating) as c FROM ratings', one=True)['c']
    }
    
    most_rated = query_db('''
        SELECT b.*, COUNT(r.id) as rating_count
        FROM books b
        JOIN ratings r ON b.id = r.book_id
        GROUP BY b.id
        ORDER BY rating_count DESC
        LIMIT 10
    ''')
    
    highest_rated = query_db('''
        SELECT b.*, AVG(r.rating) as avg_rating, COUNT(r.id) as rating_count
        FROM books b
        JOIN ratings r ON b.id = r.book_id
        GROUP BY b.id
        HAVING rating_count >= 3
        ORDER BY avg_rating DESC
        LIMIT 10
    ''')
    
    return render_template('admin/dashboard.html', stats=stats, most_rated=most_rated, highest_rated=highest_rated)

@app.route('/admin/books', methods=['GET', 'POST'])
@admin_required
def admin_books():
    if request.method == 'POST':
        title = request.form['title']
        author = request.form['author']
        genre = request.form['genre']
        description = request.form.get('description', '')
        year = request.form.get('year', '')
        cover_image = request.form.get('cover_image', '')
        
        execute_db('''
            INSERT INTO books (title, author, genre, description, year, cover_image, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', [title, author, genre, description, year, cover_image, datetime.now()])
        flash('Book added', 'success')
        return redirect(url_for('admin_books'))
    
    page = request.args.get('page', 1, type=int)
    per_page = 10
    offset = (page - 1) * per_page
    
    # Get total count
    total_count = query_db('SELECT COUNT(*) as c FROM books', one=True)['c']
    total_pages = (total_count + per_page - 1) // per_page
    
    books = query_db('''
        SELECT b.*, 
               COUNT(r.id) as rating_count,
               AVG(r.rating) as avg_rating
        FROM books b
        LEFT JOIN ratings r ON b.id = r.book_id
        GROUP BY b.id
        ORDER BY b.created_at DESC
        LIMIT ? OFFSET ?
    ''', [per_page, offset])
    
    return render_template('admin/books.html', books=books, page=page, total_pages=total_pages, total_count=total_count)

@app.route('/admin/book/<int:book_id>/edit', methods=['POST'])
@admin_required
def admin_edit_book(book_id):
    title = request.form['title']
    author = request.form['author']
    genre = request.form['genre']
    description = request.form.get('description', '')
    year = request.form.get('year', '')
    cover_image = request.form.get('cover_image', '')
    
    execute_db('''
        UPDATE books SET title = ?, author = ?, genre = ?, description = ?, year = ?, cover_image = ?
        WHERE id = ?
    ''', [title, author, genre, description, year, cover_image, book_id])
    flash('Book updated', 'success')
    return redirect(url_for('admin_books'))

@app.route('/admin/book/<int:book_id>/delete', methods=['POST'])
@admin_required
def admin_delete_book(book_id):
    execute_db('DELETE FROM ratings WHERE book_id = ?', [book_id])
    execute_db('DELETE FROM books WHERE id = ?', [book_id])
    flash('Book deleted', 'success')
    return redirect(url_for('admin_books'))

@app.route('/admin/users')
@admin_required
def admin_users():
    users = query_db('''
        SELECT u.*, 
               (SELECT COUNT(*) FROM ratings WHERE user_id = u.id) as rating_count,
               (SELECT AVG(rating) FROM ratings WHERE user_id = u.id) as avg_rating
        FROM users u
        ORDER BY u.created_at DESC
    ''')
    return render_template('admin/users.html', users=users)

@app.route('/admin/user/<int:user_id>/toggle_admin', methods=['POST'])
@admin_required
def admin_toggle_admin(user_id):
    user = query_db('SELECT * FROM users WHERE id = ?', [user_id], one=True)
    if user:
        new_status = 0 if user['is_admin'] else 1
        execute_db('UPDATE users SET is_admin = ? WHERE id = ?', [new_status, user_id])
        flash(f'Admin status updated for {user["username"]}', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/user/<int:user_id>/delete', methods=['POST'])
@admin_required
def admin_delete_user(user_id):
    if user_id == session['user_id']:
        flash('Cannot delete your own account', 'danger')
        return redirect(url_for('admin_users'))
    user = query_db('SELECT * FROM users WHERE id = ?', [user_id], one=True)
    if user:
        execute_db('DELETE FROM ratings WHERE user_id = ?', [user_id])
        execute_db('DELETE FROM users WHERE id = ?', [user_id])
        flash(f'User {user["username"]} deleted', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/analytics')
@admin_required
def admin_analytics():
    # Rating distribution
    rating_dist = query_db('''
        SELECT rating, COUNT(*) as count
        FROM ratings
        GROUP BY rating
        ORDER BY rating
    ''')
    
    # Genre distribution
    genre_dist = query_db('''
        SELECT genre, COUNT(*) as book_count,
               (SELECT COUNT(*) FROM ratings r JOIN books b2 ON r.book_id = b2.id WHERE b2.genre = b.genre) as rating_count,
               (SELECT AVG(rating) FROM ratings r JOIN books b2 ON r.book_id = b2.id WHERE b2.genre = b.genre) as avg_rating
        FROM books b
        GROUP BY genre
        ORDER BY book_count DESC
    ''')
    
    # Top raters
    top_raters = query_db('''
        SELECT u.username, COUNT(r.id) as rating_count, AVG(r.rating) as avg_rating
        FROM users u
        JOIN ratings r ON u.id = r.user_id
        GROUP BY u.id
        ORDER BY rating_count DESC
        LIMIT 10
    ''')
    
    # Recent activity
    recent_ratings = query_db('''
        SELECT r.*, u.username, b.title
        FROM ratings r
        JOIN users u ON r.user_id = u.id
        JOIN books b ON r.book_id = b.id
        ORDER BY r.created_at DESC
        LIMIT 20
    ''')
    
    return render_template('admin/analytics.html', 
                          rating_dist=rating_dist,
                          genre_dist=genre_dist,
                          top_raters=top_raters,
                          recent_ratings=recent_ratings)

@app.route('/admin/recommendations')
@admin_required
def admin_recommendations():
    # Algorithm settings stored in session for demo
    algo_settings = {
        'method': session.get('rec_method', 'cosine'),
        'min_similarity': session.get('min_similarity', 0.1),
        'min_common_ratings': session.get('min_common_ratings', 2),
        'top_n': session.get('rec_top_n', 5)
    }
    
    # Test recommendation for a sample user
    test_user_id = request.args.get('test_user', 1, type=int)
    test_recommendations = RecommendationEngine.recommend_books(
        test_user_id, n=6, method=algo_settings['method']
    )
    
    # Similarity matrix sample
    user_ratings, _, _ = RecommendationEngine.get_user_ratings_matrix()
    sample_users = list(user_ratings.keys())[:10] if user_ratings else []
    
    return render_template('admin/recommendations.html',
                          algo_settings=algo_settings,
                          test_recommendations=test_recommendations,
                          test_user_id=test_user_id,
                          sample_users=sample_users,
                          user_ratings=user_ratings)

@app.route('/admin/recommendations/settings', methods=['POST'])
@admin_required
def admin_rec_settings():
    session['rec_method'] = request.form.get('method', 'cosine')
    session['min_similarity'] = float(request.form.get('min_similarity', 0.1))
    session['min_common_ratings'] = int(request.form.get('min_common_ratings', 2))
    session['rec_top_n'] = int(request.form.get('top_n', 5))
    flash('Recommendation settings updated', 'success')
    return redirect(url_for('admin_recommendations'))

@app.route('/admin/precompute', methods=['POST'])
@admin_required
def admin_precompute():
    # Precompute recommendations for all users
    users = query_db('SELECT id FROM users')
    precomputed = {}
    for user in users:
        recs = RecommendationEngine.recommend_books(user['id'], n=10)
        precomputed[user['id']] = recs
    
    # Store in session for demo (in production, store in database/cache)
    session['precomputed_recommendations'] = precomputed
    flash(f'Precomputed recommendations for {len(users)} users', 'success')
    return redirect(url_for('admin_recommendations'))

@app.route('/api/book/<int:book_id>/ratings')
def api_book_ratings(book_id):
    ratings = query_db('''
        SELECT r.*, u.username
        FROM ratings r
        JOIN users u ON r.user_id = u.id
        WHERE r.book_id = ?
        ORDER BY r.created_at DESC
    ''', [book_id])
    return jsonify([dict(r) for r in ratings])

@app.route('/admin/rating/<int:rating_id>/delete', methods=['POST'])
@admin_required
def admin_delete_rating(rating_id):
    rating = query_db('SELECT * FROM ratings WHERE id = ?', [rating_id], one=True)
    if rating:
        execute_db('DELETE FROM ratings WHERE id = ?', [rating_id])
        flash('Rating deleted', 'success')
    return redirect(url_for('admin_books'))

@app.route('/api/recommendations')
@login_required
def api_recommendations():
    n = request.args.get('n', 5, type=int)
    recommendations = RecommendationEngine.recommend_books(session['user_id'], n)
    return jsonify(recommendations)

if __name__ == '__main__':
    if not os.path.exists(app.config['DATABASE']):
        init_db()
    app.run(debug=True, host='0.0.0.0', port=5001)
