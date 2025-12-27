import sqlite3
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TwitterDatabase:
    def __init__(self, db_path: str = "twitter_analyzer.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                
                # Tweets table - stores all tweets
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS tweets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user TEXT NOT NULL,
                        content TEXT NOT NULL,
                        likes INTEGER DEFAULT 0,
                        retweets INTEGER DEFAULT 0,
                        replies INTEGER DEFAULT 0,
                        quotes INTEGER DEFAULT 0,
                        bookmarks INTEGER DEFAULT 0,
                        views INTEGER DEFAULT 0,
                        engagement_score REAL DEFAULT 0,
                        content_length INTEGER DEFAULT 0,
                        created_at TEXT,
                        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        dataset_id TEXT NOT NULL
                    )
                """)
                
                # Personas table - stores analyzed persona data
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS personas (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        dataset_id TEXT NOT NULL,
                        total_tweets INTEGER DEFAULT 0,
                        writing_patterns TEXT, -- JSON
                        ai_analysis TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Generated tweets table - stores AI generated tweets
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS generated_tweets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        persona_id INTEGER,
                        context TEXT NOT NULL,
                        content TEXT NOT NULL,
                        predicted_engagement TEXT,
                        style_similarity TEXT,
                        reasoning TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (persona_id) REFERENCES personas (id)
                    )
                """)
                
                # Create indexes for better performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_tweets_dataset ON tweets(dataset_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_tweets_engagement ON tweets(engagement_score DESC)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_personas_name ON personas(name)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_generated_persona ON generated_tweets(persona_id)")
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    def store_tweets(self, df: pd.DataFrame, dataset_id: str) -> int:
        """Store tweets from DataFrame into database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Convert DataFrame to records
                records = []
                for _, row in df.iterrows():
                    record = (
                        row.get('user', ''),
                        row.get('content', ''),
                        int(row.get('likes', 0)),
                        int(row.get('retweets', 0)),
                        int(row.get('replies', 0)),
                        int(row.get('quotes', 0)),
                        int(row.get('bookmarks', 0)),
                        int(row.get('views', 0)),
                        float(row.get('engagement_score', 0)),
                        len(str(row.get('content', ''))),
                        row.get('created_at', ''),
                        dataset_id
                    )
                    records.append(record)
                
                conn.executemany("""
                    INSERT INTO tweets (
                        user, content, likes, retweets, replies, quotes, 
                        bookmarks, views, engagement_score, content_length, 
                        created_at, dataset_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, records)
                
                conn.commit()
                logger.info(f"Stored {len(records)} tweets for dataset {dataset_id}")
                return len(records)
                
        except sqlite3.Error as e:
            logger.error(f"Error storing tweets: {e}")
            raise
    
    def get_tweets(self, dataset_id: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Retrieve tweets for a specific dataset"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM tweets WHERE dataset_id = ?"
                params = [dataset_id]
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                df = pd.read_sql_query(query, conn, params=params)
                logger.info(f"Retrieved {len(df)} tweets for dataset {dataset_id}")
                return df
                
        except sqlite3.Error as e:
            logger.error(f"Error retrieving tweets: {e}")
            raise
    
    def get_top_performers(self, dataset_id: str, top_n: int = 20) -> pd.DataFrame:
        """Get top performing tweets for a dataset"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT * FROM tweets 
                    WHERE dataset_id = ? 
                    ORDER BY engagement_score DESC 
                    LIMIT ?
                """
                df = pd.read_sql_query(query, conn, params=[dataset_id, top_n])
                logger.info(f"Retrieved {len(df)} top performers for dataset {dataset_id}")
                return df
                
        except sqlite3.Error as e:
            logger.error(f"Error retrieving top performers: {e}")
            raise
    
    def store_persona(self, name: str, dataset_id: str, total_tweets: int, 
                     writing_patterns: Dict[str, Any], ai_analysis: str) -> int:
        """Store persona analysis data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if persona exists
                cursor.execute("SELECT id FROM personas WHERE name = ?", (name,))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing persona
                    cursor.execute("""
                        UPDATE personas 
                        SET dataset_id = ?, total_tweets = ?, writing_patterns = ?, 
                            ai_analysis = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE name = ?
                    """, (dataset_id, total_tweets, json.dumps(writing_patterns), 
                          ai_analysis, name))
                    persona_id = existing[0]
                    logger.info(f"Updated persona {name}")
                else:
                    # Insert new persona
                    cursor.execute("""
                        INSERT INTO personas (name, dataset_id, total_tweets, writing_patterns, ai_analysis)
                        VALUES (?, ?, ?, ?, ?)
                    """, (name, dataset_id, total_tweets, json.dumps(writing_patterns), ai_analysis))
                    persona_id = cursor.lastrowid
                    logger.info(f"Created new persona {name}")
                
                conn.commit()
                return persona_id
                
        except sqlite3.Error as e:
            logger.error(f"Error storing persona: {e}")
            raise
    
    def get_persona(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve persona data by name"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM personas WHERE name = ?", (name,))
                row = cursor.fetchone()
                
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    persona_data = dict(zip(columns, row))
                    
                    # Parse JSON fields
                    if persona_data['writing_patterns']:
                        persona_data['writing_patterns'] = json.loads(persona_data['writing_patterns'])
                    
                    logger.info(f"Retrieved persona {name}")
                    return persona_data
                
                return None
                
        except sqlite3.Error as e:
            logger.error(f"Error retrieving persona: {e}")
            raise
    
    def store_generated_tweet(self, persona_id: int, context: str, content: str,
                            predicted_engagement: str, style_similarity: str, reasoning: str) -> int:
        """Store a generated tweet"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO generated_tweets 
                    (persona_id, context, content, predicted_engagement, style_similarity, reasoning)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (persona_id, context, content, predicted_engagement, style_similarity, reasoning))
                
                conn.commit()
                tweet_id = cursor.lastrowid
                logger.info(f"Stored generated tweet {tweet_id}")
                return tweet_id
                
        except sqlite3.Error as e:
            logger.error(f"Error storing generated tweet: {e}")
            raise
    
    def get_generated_tweets(self, persona_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Get generated tweets for a persona"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM generated_tweets 
                    WHERE persona_id = ? 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (persona_id, limit))
                
                columns = [desc[0] for desc in cursor.description]
                tweets = [dict(zip(columns, row)) for row in cursor.fetchall()]
                
                logger.info(f"Retrieved {len(tweets)} generated tweets for persona {persona_id}")
                return tweets
                
        except sqlite3.Error as e:
            logger.error(f"Error retrieving generated tweets: {e}")
            raise
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count tweets
                cursor.execute("SELECT COUNT(*) FROM tweets")
                total_tweets = cursor.fetchone()[0]
                
                # Count personas
                cursor.execute("SELECT COUNT(*) FROM personas")
                total_personas = cursor.fetchone()[0]
                
                # Count generated tweets
                cursor.execute("SELECT COUNT(*) FROM generated_tweets")
                total_generated = cursor.fetchone()[0]
                
                # Get datasets
                cursor.execute("SELECT DISTINCT dataset_id FROM tweets")
                datasets = [row[0] for row in cursor.fetchall()]
                
                return {
                    "total_tweets": total_tweets,
                    "total_personas": total_personas,
                    "total_generated_tweets": total_generated,
                    "datasets": datasets
                }
                
        except sqlite3.Error as e:
            logger.error(f"Error getting database stats: {e}")
            raise
    
    def clear_dataset(self, dataset_id: str):
        """Clear all data for a specific dataset"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Delete tweets
                conn.execute("DELETE FROM tweets WHERE dataset_id = ?", (dataset_id,))
                
                # Delete personas for this dataset
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM personas WHERE dataset_id = ?", (dataset_id,))
                persona_ids = [row[0] for row in cursor.fetchall()]
                
                if persona_ids:
                    placeholders = ','.join(['?'] * len(persona_ids))
                    conn.execute(f"DELETE FROM generated_tweets WHERE persona_id IN ({placeholders})", persona_ids)
                    conn.execute("DELETE FROM personas WHERE dataset_id = ?", (dataset_id,))
                
                conn.commit()
                logger.info(f"Cleared all data for dataset {dataset_id}")
                
        except sqlite3.Error as e:
            logger.error(f"Error clearing dataset: {e}")
            raise

# Global database instance
db = TwitterDatabase() 