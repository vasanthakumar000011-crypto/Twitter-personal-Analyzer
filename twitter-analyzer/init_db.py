#!/usr/bin/env python3
"""
Database initialization script for Twitter Analyzer
Run this to set up the SQLite database
"""

import os
import sys
from pathlib import Path

# Add the current directory to the path so we can import the database module
sys.path.append(str(Path(__file__).parent))

from database import TwitterDatabase

def init_database():
    """Initialize the database"""
    print("Initializing Twitter Analyzer database...")
    
    # Create database instance
    db = TwitterDatabase("twitter_analyzer.db")
    
    # Get initial stats
    stats = db.get_database_stats()
    
    print(f"✅ Database initialized successfully!")
    print(f"📊 Database Statistics:")
    print(f"   - Total tweets: {stats['total_tweets']}")
    print(f"   - Total personas: {stats['total_personas']}")
    print(f"   - Total generated tweets: {stats['total_generated_tweets']}")
    print(f"   - Datasets: {stats['datasets']}")
    
    print(f"\n📁 Database file: {os.path.abspath('twitter_analyzer.db')}")
    print("🚀 Ready to use! Start the server with: python main.py")

if __name__ == "__main__":
    init_database() 