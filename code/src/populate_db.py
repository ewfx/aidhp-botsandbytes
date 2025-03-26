from app import app, db, Content
import json

# Sample content data
sample_content = [
    {
        "title": "Introduction to Machine Learning",
        "category": "Technology",
        "tags": ["AI", "programming", "data science", "technology"]
    },
    {
        "title": "Modern Web Development Trends",
        "category": "Technology",
        "tags": ["web development", "programming", "javascript", "technology"]
    },
    {
        "title": "Healthy Mediterranean Recipes",
        "category": "Food",
        "tags": ["cooking", "health", "food", "mediterranean"]
    },
    {
        "title": "Digital Photography Basics",
        "category": "Photography",
        "tags": ["photography", "art", "creativity", "technology"]
    },
    {
        "title": "Sustainable Living Guide",
        "category": "Lifestyle",
        "tags": ["environment", "sustainability", "lifestyle", "health"]
    },
    {
        "title": "Home Workout Routines",
        "category": "Fitness",
        "tags": ["fitness", "health", "exercise", "lifestyle"]
    },
    {
        "title": "Financial Planning 101",
        "category": "Finance",
        "tags": ["finance", "money", "planning", "education"]
    },
    {
        "title": "Art of Coffee Making",
        "category": "Food",
        "tags": ["coffee", "food", "beverages", "cooking"]
    },
    {
        "title": "Travel Photography Tips",
        "category": "Photography",
        "tags": ["photography", "travel", "art", "creativity"]
    },
    {
        "title": "Data Science Career Guide",
        "category": "Career",
        "tags": ["data science", "career", "technology", "education"]
    }
]

def populate_database():
    with app.app_context():
        # Clear existing content
        Content.query.delete()
        
        # Add sample content
        for item in sample_content:
            content = Content(
                title=item["title"],
                category=item["category"],
                tags=json.dumps(item["tags"])
            )
            db.session.add(content)
        
        # Commit changes
        db.session.commit()
        print("Database populated with sample content!")

if __name__ == "__main__":
    populate_database()
