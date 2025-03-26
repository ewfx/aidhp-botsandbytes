import pandas as pd
import json
from app import db, User, Content, Interaction
from datetime import datetime

def import_users_from_excel(file_path):
    """
    Import users from Excel file with the following columns:
    - Customer ID
    - Age
    - Gender
    - Purchase History
    - Interests
    - Engagement score (0-200)
    - Sentiment Score (-1 to 1)
    - Social media activity level (Low/Med/High)
    """
    try:
        df = pd.read_excel(file_path)
        
        # Validate required columns
        required_columns = [
            'Customer ID', 'Age', 'Gender', 'Purchase History',
            'Interests', 'Engagement score', 'Sentiment Score',
            'Social media activity level'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Process each row
        for _, row in df.iterrows():
            # Clean and validate data
            customer_id = str(row['Customer ID'])
            age = int(row['Age']) if pd.notna(row['Age']) else 0
            gender = str(row['Gender']) if pd.notna(row['Gender']) else ''
            
            # Convert purchase history to JSON
            purchase_history = row['Purchase History']
            if isinstance(purchase_history, str):
                purchase_history = [p.strip() for p in purchase_history.split(',') if p.strip()]
            else:
                purchase_history = []
            
            # Convert interests to JSON
            interests = row['Interests']
            if isinstance(interests, str):
                interests = [i.strip() for i in interests.split(',') if i.strip()]
            else:
                interests = []
            
            # Validate and clean scores
            engagement_score = float(row['Engagement score'])
            engagement_score = max(0, min(200, engagement_score))  # Clamp between 0-200
            
            sentiment_score = float(row['Sentiment Score'])
            sentiment_score = max(-1, min(1, sentiment_score))  # Clamp between -1 and 1
            
            # Normalize social media activity
            social_media = str(row['Social media activity level']).strip().capitalize()
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
        
        db.session.commit()
        return True, f"Successfully imported {len(df)} users"
    except Exception as e:
        return False, f"Error importing users: {str(e)}"

def import_content_from_excel(file_path):
    """
    Import content from Excel file.
    Expected columns: title, category, tags
    """
    try:
        df = pd.read_excel(file_path, sheet_name='Content')
        for _, row in df.iterrows():
            title = row['title']
            category = row['category']
            tags = row['tags'].split(',') if isinstance(row['tags'], str) else []
            
            # Create or update content
            content = Content.query.filter_by(title=title).first()
            if not content:
                content = Content(title=title)
                db.session.add(content)
            
            content.category = category
            content.tags = f"[{','.join(f'\"{tag.strip()}\"' for tag in tags)}]"
        
        db.session.commit()
        return True, f"Successfully imported {len(df)} content items"
    except Exception as e:
        return False, f"Error importing content: {str(e)}"

def import_interactions_from_excel(file_path):
    """
    Import interactions from Excel file.
    Expected columns: username, content_title, interaction_type, timestamp
    """
    try:
        df = pd.read_excel(file_path, sheet_name='Interactions')
        for _, row in df.iterrows():
            username = row['username']
            content_title = row['content_title']
            interaction_type = row['interaction_type']
            timestamp = row['timestamp'] if 'timestamp' in row else datetime.now()
            
            # Get user and content
            user = User.query.filter_by(username=username).first()
            content = Content.query.filter_by(title=content_title).first()
            
            if user and content:
                # Create interaction
                interaction = Interaction(
                    user_id=user.id,
                    content_id=content.id,
                    interaction_type=interaction_type,
                    timestamp=timestamp
                )
                db.session.add(interaction)
        
        db.session.commit()
        return True, f"Successfully imported {len(df)} interactions"
    except Exception as e:
        return False, f"Error importing interactions: {str(e)}"
