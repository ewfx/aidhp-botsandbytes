import sqlite3
import os

def init_db():
    # Create instance directory if it doesn't exist
    os.makedirs('instance', exist_ok=True)
    
    # Connect to SQLite database
    conn = sqlite3.connect('instance/database.db')
    cursor = conn.cursor()
    
    try:
        # Create user table with new structure
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT UNIQUE NOT NULL,
                age INTEGER,
                gender TEXT,
                purchase_history TEXT,
                interests TEXT,
                engagement_score REAL,
                sentiment_score REAL,
                social_media_activity TEXT
            )
        ''')
        
        # Create content table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT UNIQUE NOT NULL,
                category TEXT,
                tags TEXT
            )
        ''')
        
        # Create interaction table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interaction (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                content_id INTEGER,
                interaction_type TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user (id),
                FOREIGN KEY (content_id) REFERENCES content (id)
            )
        ''')
        
        conn.commit()
        print("Database initialized successfully")
        
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == '__main__':
    init_db()
