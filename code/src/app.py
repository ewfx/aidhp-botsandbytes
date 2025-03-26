import os
import json
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os
from werkzeug.utils import secure_filename

# Initialize Flask app and extensions
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Change in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)

# Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.String(50), unique=True)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    purchase_history = db.Column(db.Text)  # JSON string
    interests = db.Column(db.Text)  # JSON string
    engagement_score = db.Column(db.Float)
    sentiment_score = db.Column(db.Float)
    social_media_activity = db.Column(db.String(10))
    interactions = db.relationship('Interaction', backref='user', lazy=True)

    @staticmethod
    def get_default_user():
        user = User.query.first()
        if not user:
            user = User(
                customer_id="default_user",
                age=0,
                gender="",
                purchase_history="[]",
                interests="[]",
                engagement_score=0,
                sentiment_score=0,
                social_media_activity="Low"
            )
            db.session.add(user)
            db.session.commit()
        return user

class Content(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(50))
    tags = db.Column(db.Text)
    interactions = db.relationship('Interaction', backref='content', lazy=True)

class Interaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    interaction_type = db.Column(db.String(20))  # view, like, share
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Create database tables
def add_sample_data():
    try:
        # Add a sample user
        sample_user = User(
            customer_id='SAMPLE001',
            age=30,
            gender='Male',
            purchase_history=json.dumps(['Electronics', 'Books']),
            interests=json.dumps(['Technology', 'Reading']),
            engagement_score=150.0,
            sentiment_score=0.8,
            social_media_activity='High'
        )
        db.session.add(sample_user)
        db.session.commit()
        print("Added sample user successfully")
    except Exception as e:
        print(f"Error adding sample data: {str(e)}")
        db.session.rollback()

with app.app_context():
    db.drop_all()
    db.create_all()
    add_sample_data()

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Add file upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def import_users_from_excel(filepath):
    try:
        df = pd.read_excel(filepath)
        for index, row in df.iterrows():
            user = User.query.filter_by(customer_id=row['customer_id']).first()
            if not user:
                user = User(
                    customer_id=row['customer_id'],
                    age=row['age'],
                    gender=row['gender'],
                    purchase_history=row['purchase_history'],
                    interests=row['interests'],
                    engagement_score=row['engagement_score'],
                    sentiment_score=row['sentiment_score'],
                    social_media_activity=row['social_media_activity']
                )
                db.session.add(user)
        db.session.commit()
        return True, "Users imported successfully"
    except Exception as e:
        return False, str(e)

def import_content_from_excel(filepath):
    try:
        df = pd.read_excel(filepath)
        for index, row in df.iterrows():
            content = Content.query.filter_by(title=row['title']).first()
            if not content:
                content = Content(title=row['title'], category=row['category'], tags=row['tags'])
                db.session.add(content)
        db.session.commit()
        return True, "Content imported successfully"
    except Exception as e:
        return False, str(e)

def import_interactions_from_excel(filepath):
    try:
        df = pd.read_excel(filepath)
        for index, row in df.iterrows():
            interaction = Interaction.query.filter_by(user_id=row['user_id'], content_id=row['content_id'], interaction_type=row['interaction_type']).first()
            if not interaction:
                interaction = Interaction(user_id=row['user_id'], content_id=row['content_id'], interaction_type=row['interaction_type'])
                db.session.add(interaction)
        db.session.commit()
        return True, "Interactions imported successfully"
    except Exception as e:
        return False, str(e)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/profile', methods=['GET', 'POST'])
def profile():
    user = User.get_default_user()
    if request.method == 'POST':
        data = request.json
        user.purchase_history = json.dumps(data.get('purchase_history', []))
        user.interests = json.dumps(data.get('interests', []))
        user.engagement_score = data.get('engagement_score', 0)
        user.sentiment_score = data.get('sentiment_score', 0)
        user.social_media_activity = data.get('social_media_activity', "Low")
        db.session.commit()
        return jsonify({'status': 'success'})
    return jsonify({
        'purchase_history': json.loads(user.purchase_history),
        'interests': json.loads(user.interests),
        'engagement_score': user.engagement_score,
        'sentiment_score': user.sentiment_score,
        'social_media_activity': user.social_media_activity
    })

@app.route('/api/recommend')
def get_recommendations():
    try:
        # Get user_id from query parameter
        user_id = request.args.get('user_id')
        
        if user_id:
            # Get specific user
            user = User.query.get(int(user_id))
            if not user:
                return jsonify({
                    'recommendations': [],
                    'message': f'User {user_id} not found'
                })
            users = [user]  # Generate recommendations only for this user
        else:
            # Get all users if no specific user requested
            users = User.query.all()
        
        print(f"Generating recommendations for {len(users)} users")
        
        if not users:
            return jsonify({
                'recommendations': [],
                'message': 'No user data available. Please upload user data first.'
            })

        # Generate recommendations based on user interests
        recommendations = []
        for user in users:
            try:
                interests = json.loads(user.interests) if user.interests else []
                purchase_history = json.loads(user.purchase_history) if user.purchase_history else []
                
                print(f"User {user.customer_id} - Interests: {interests}, History: {purchase_history}")
                
                # Generate recommendations based on interests and purchase history
                if interests:
                    recommendations.append({
                        'id': len(recommendations) + 1,
                        'title': f'Recommended based on interest in {interests[0]}',
                        'description': f'This matches your interest in {", ".join(interests)}',
                        'confidence': 0.95,
                        'user_id': user.id
                    })
                
                if purchase_history:
                    recommendations.append({
                        'id': len(recommendations) + 1,
                        'title': f'Similar to your purchase: {purchase_history[0]}',
                        'description': f'Based on your purchase history: {", ".join(purchase_history)}',
                        'confidence': 0.90,
                        'user_id': user.id
                    })
                
                # Add recommendation based on engagement score
                if user.engagement_score > 100:
                    recommendations.append({
                        'id': len(recommendations) + 1,
                        'title': 'Premium Product Recommendation',
                        'description': 'Based on your high engagement score',
                        'confidence': 0.85,
                        'user_id': user.id
                    })
                
                # Add recommendation based on sentiment score
                if user.sentiment_score > 0.5:
                    recommendations.append({
                        'id': len(recommendations) + 1,
                        'title': 'Exclusive Offer',
                        'description': 'Special recommendation for our happy customers',
                        'confidence': 0.88,
                        'user_id': user.id
                    })
                
                # Add recommendation based on social media activity
                if user.social_media_activity == 'High':
                    recommendations.append({
                        'id': len(recommendations) + 1,
                        'title': 'Social Media Exclusive',
                        'description': 'Perfect for sharing with your network',
                        'confidence': 0.92,
                        'user_id': user.id
                    })
                
            except Exception as e:
                print(f"Error generating recommendations for user {user.customer_id}: {str(e)}")
                continue
        
        # Filter recommendations for specific user if requested
        if user_id:
            recommendations = [r for r in recommendations if r['user_id'] == int(user_id)]
        
        print(f"Generated {len(recommendations)} recommendations")
        
        return jsonify({
            'recommendations': recommendations[:5],  # Return top 5 recommendations
            'message': f'Generated {len(recommendations)} recommendations based on {len(users)} user profiles'
        })

    except Exception as e:
        print(f"Error in recommendations: {str(e)}")
        return jsonify({
            'recommendations': [],
            'message': f'Error generating recommendations: {str(e)}'
        })

@app.route('/api/interact', methods=['POST'])
def record_interaction():
    data = request.json
    interaction = Interaction(
        user_id=User.get_default_user().id,
        content_id=data['content_id'],
        interaction_type=data['type']
    )
    db.session.add(interaction)
    db.session.commit()
    return jsonify({'status': 'success'})

@app.route('/download/template')
def download_template():
    try:
        return send_file('user_data_template.xlsx',
                        as_attachment=True,
                        download_name='user_data_template.xlsx')
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.xlsx'):
        return jsonify({'error': 'Invalid file format. Please upload an Excel file (.xlsx)'}), 400
    
    try:
        # Save the file temporarily
        temp_path = 'temp_upload.xlsx'
        file.save(temp_path)
        
        # Import users from Excel
        df = pd.read_excel(temp_path)
        print("Found columns:", df.columns.tolist())
        
        # Clean column names by stripping whitespace
        df.columns = df.columns.str.strip()
        
        # Column name mappings (case-insensitive)
        column_mappings = {
            'customer_id': ['Customer ID'],
            'age': ['Age'],
            'gender': ['Gender'],
            'purchase_history': ['Purchase History'],
            'interests': ['Interests'],
            'engagement_score': ['Engagement score (0-200)'],
            'sentiment_score': ['Sentiment Score (-1 to 1)'],
            'social_media_activity': ['Social media activity level (Low/Med/High)']
        }
        
        # Find actual column names in the DataFrame
        actual_columns = {}
        for key, variations in column_mappings.items():
            found = False
            for var in variations:
                if var.strip() in df.columns:
                    actual_columns[key] = var.strip()
                    found = True
                    break
            if not found:
                print(f"Missing column {key}, variations tried: {variations}")
        
        # Check for missing required columns
        missing_columns = []
        for key in column_mappings.keys():
            if key not in actual_columns:
                missing_columns.append(key)
        
        if missing_columns:
            os.remove(temp_path)
            return jsonify({'error': f'Missing required columns: {", ".join(missing_columns)}. Available columns: {", ".join(df.columns.tolist())}'}), 400
        
        # Process each row
        users_added = 0
        for _, row in df.iterrows():
            try:
                # Clean and validate data
                customer_id = str(row[actual_columns['customer_id']])
                age = int(float(row[actual_columns['age']])) if pd.notna(row[actual_columns['age']]) else 0
                gender = str(row[actual_columns['gender']]) if pd.notna(row[actual_columns['gender']]) else ''
                
                # Convert purchase history to JSON
                purchase_history = row[actual_columns['purchase_history']]
                if isinstance(purchase_history, str):
                    purchase_history = [p.strip() for p in purchase_history.split(',') if p.strip()]
                else:
                    purchase_history = []
                
                # Convert interests to JSON
                interests = row[actual_columns['interests']]
                if isinstance(interests, str):
                    interests = [i.strip() for i in interests.split(',') if i.strip()]
                else:
                    interests = []
                
                # Validate and clean scores
                engagement_score = float(row[actual_columns['engagement_score']])
                engagement_score = max(0, min(200, engagement_score))
                
                sentiment_score = float(row[actual_columns['sentiment_score']])
                sentiment_score = max(-1, min(1, sentiment_score))
                
                # Normalize social media activity
                social_media = str(row[actual_columns['social_media_activity']]).strip().capitalize()
                if social_media not in ['Low', 'Med', 'High']:
                    social_media = 'Low'
                
                # Create or update user
                user = User.query.filter_by(customer_id=customer_id).first()
                if not user:
                    user = User(customer_id=customer_id)
                    db.session.add(user)
                
                # Update user data
                user.age = age
                user.gender = gender
                user.purchase_history = json.dumps(purchase_history)
                user.interests = json.dumps(interests)
                user.engagement_score = engagement_score
                user.sentiment_score = sentiment_score
                user.social_media_activity = social_media
                
                users_added += 1
                print(f"Added/Updated user {customer_id}")
                
            except Exception as e:
                print(f"Error processing row: {str(e)}")
                continue
        
        db.session.commit()
        print(f"Successfully imported {users_added} users")
        
        # Clean up temp file
        os.remove(temp_path)
        return jsonify({'message': f'Successfully imported {users_added} users'}), 200
            
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        # Clean up temp file in case of error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/api/upload', methods=['POST'])
def upload_excel():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        results = []
        
        # Import data based on file type
        if 'users' in filename.lower():
            success, message = import_users_from_excel(filepath)
            results.append({'type': 'users', 'success': success, 'message': message})
        
        if 'content' in filename.lower():
            success, message = import_content_from_excel(filepath)
            results.append({'type': 'content', 'success': success, 'message': message})
        
        if 'interactions' in filename.lower():
            success, message = import_interactions_from_excel(filepath)
            results.append({'type': 'interactions', 'success': success, 'message': message})
        
        # Clean up the uploaded file
        os.remove(filepath)
        
        return jsonify({'results': results})
    
    return jsonify({'error': 'Invalid file type'}), 400

# Get all users
@app.route('/api/users', methods=['GET'])
def get_users():
    try:
        users = User.query.all()
        return jsonify([{
            'id': user.id,
            'customer_id': user.customer_id,
            'age': user.age,
            'gender': user.gender,
            'interests': json.loads(user.interests) if user.interests else [],
            'purchase_history': json.loads(user.purchase_history) if user.purchase_history else [],
            'engagement_score': user.engagement_score,
            'sentiment_score': user.sentiment_score,
            'social_media_activity': user.social_media_activity
        } for user in users])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Get single user
@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
            
        return jsonify({
            'id': user.id,
            'customer_id': user.customer_id,
            'age': user.age,
            'gender': user.gender,
            'interests': json.loads(user.interests) if user.interests else [],
            'purchase_history': json.loads(user.purchase_history) if user.purchase_history else [],
            'engagement_score': user.engagement_score,
            'sentiment_score': user.sentiment_score,
            'social_media_activity': user.social_media_activity
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Update user
@app.route('/api/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
            
        data = request.json
        if 'age' in data:
            user.age = int(data['age'])
        if 'gender' in data:
            user.gender = data['gender']
        if 'interests' in data:
            user.interests = json.dumps(data['interests'])
        if 'purchase_history' in data:
            user.purchase_history = json.dumps(data['purchase_history'])
        if 'engagement_score' in data:
            user.engagement_score = float(data['engagement_score'])
        if 'sentiment_score' in data:
            user.sentiment_score = float(data['sentiment_score'])
        if 'social_media_activity' in data:
            user.social_media_activity = data['social_media_activity']
            
        db.session.commit()
        return jsonify({'message': 'User updated successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Delete user
@app.route('/api/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
            
        db.session.delete(user)
        db.session.commit()
        return jsonify({'message': 'User deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
