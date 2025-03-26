import sqlite3
import os

def upgrade():
    # Connect to SQLite database
    conn = sqlite3.connect('instance/database.db')
    cursor = conn.cursor()
    
    try:
        # Add new columns
        cursor.execute('ALTER TABLE user ADD COLUMN customer_id TEXT DEFAULT "default"')
        cursor.execute('ALTER TABLE user ADD COLUMN age INTEGER DEFAULT 0')
        cursor.execute('ALTER TABLE user ADD COLUMN gender TEXT DEFAULT ""')
        cursor.execute('ALTER TABLE user ADD COLUMN purchase_history TEXT DEFAULT "[]"')
        cursor.execute('ALTER TABLE user ADD COLUMN engagement_score REAL DEFAULT 0')
        cursor.execute('ALTER TABLE user ADD COLUMN sentiment_score REAL DEFAULT 0')
        cursor.execute('ALTER TABLE user ADD COLUMN social_media_activity TEXT DEFAULT "Low"')
        
        # Create temporary table with new structure
        cursor.execute('''
            CREATE TABLE user_new (
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
        
        # Copy data from old table to new table
        cursor.execute('''
            INSERT INTO user_new (id, customer_id, interests)
            SELECT id, COALESCE(customer_id, "default"), preferences
            FROM user
        ''')
        
        # Drop old table and rename new table
        cursor.execute('DROP TABLE user')
        cursor.execute('ALTER TABLE user_new RENAME TO user')
        
        conn.commit()
        print("Database migration completed successfully")
        
    except Exception as e:
        print(f"Error during migration: {str(e)}")
        conn.rollback()
    finally:
        conn.close()

def downgrade():
    # Revert changes if needed
    with sqlite3.connect('instance/database.db') as conn:
        cursor = conn.cursor()
        try:
            # Drop new columns
            cursor.execute('''
                ALTER TABLE user
                DROP COLUMN customer_id,
                DROP COLUMN age,
                DROP COLUMN gender,
                DROP COLUMN purchase_history,
                DROP COLUMN engagement_score,
                DROP COLUMN sentiment_score,
                DROP COLUMN social_media_activity;
                
                ALTER TABLE user
                ADD COLUMN username TEXT UNIQUE NOT NULL DEFAULT 'default',
                ADD COLUMN preferences TEXT DEFAULT '{}';
            ''')
            conn.commit()
            print("Database downgrade completed successfully")
        except Exception as e:
            print(f"Error during downgrade: {str(e)}")
            conn.rollback()

if __name__ == '__main__':
    # Create instance directory if it doesn't exist
    os.makedirs('instance', exist_ok=True)
    upgrade()
