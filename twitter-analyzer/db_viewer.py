#!/usr/bin/env python3
"""
Database viewer script for Twitter Analyzer
View and manage database contents
"""

import sqlite3
import sys
from pathlib import Path
import json
from datetime import datetime

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from database import TwitterDatabase

def view_database_contents():
    """View database contents"""
    db = TwitterDatabase("twitter_analyzer.db")
    
    print("🗄️  Twitter Analyzer Database Viewer")
    print("=" * 50)
    
    # Get stats
    stats = db.get_database_stats()
    print(f"📊 Database Statistics:")
    print(f"   • Total tweets: {stats['total_tweets']}")
    print(f"   • Total personas: {stats['total_personas']}")
    print(f"   • Total generated tweets: {stats['total_generated_tweets']}")
    print(f"   • Datasets: {', '.join(stats['datasets']) if stats['datasets'] else 'None'}")
    print()
    
    # Show personas
    if stats['total_personas'] > 0:
        print("👤 Personas:")
        persona = db.get_persona("build_in_public")
        if persona:
            print(f"   • {persona['name']}")
            print(f"     - Dataset ID: {persona['dataset_id']}")
            print(f"     - Total tweets: {persona['total_tweets']}")
            print(f"     - Created: {persona['created_at']}")
            print(f"     - Updated: {persona['updated_at']}")
        print()
    
    # Show recent generated tweets
    if stats['total_generated_tweets'] > 0:
        print("🤖 Recent Generated Tweets:")
        persona = db.get_persona("build_in_public")
        if persona:
            tweets = db.get_generated_tweets(persona['id'], 5)
            for i, tweet in enumerate(tweets, 1):
                print(f"   {i}. {tweet['content'][:60]}...")
                print(f"      Context: {tweet['context'][:40]}...")
                print(f"      Engagement: {tweet['predicted_engagement']} | Similarity: {tweet['style_similarity']}")
                print(f"      Created: {tweet['created_at']}")
                print()

def show_top_performers():
    """Show top performing tweets"""
    db = TwitterDatabase("twitter_analyzer.db")
    
    # Get the first available dataset
    stats = db.get_database_stats()
    if not stats['datasets']:
        print("No datasets available")
        return
    
    dataset_id = stats['datasets'][0]
    top_tweets = db.get_top_performers(dataset_id, 10)
    
    print(f"🏆 Top 10 Performing Tweets (Dataset: {dataset_id}):")
    print("=" * 70)
    
    for i, (_, tweet) in enumerate(top_tweets.iterrows(), 1):
        print(f"{i}. Score: {tweet['engagement_score']:.1f}")
        print(f"   Content: {tweet['content'][:80]}...")
        print(f"   👍 {tweet['likes']} | 🔄 {tweet['retweets']} | 💬 {tweet['replies']}")
        print()

def interactive_menu():
    """Interactive menu for database operations"""
    while True:
        print("\n🗄️ Twitter Analyzer Database Menu")
        print("1. View database overview")
        print("2. Show top performing tweets")
        print("3. Clear all data (⚠️ DANGER)")
        print("4. Exit")
        
        choice = input("\nChoose an option (1-4): ").strip()
        
        if choice == "1":
            view_database_contents()
        elif choice == "2":
            show_top_performers()
        elif choice == "3":
            confirm = input("⚠️ This will delete ALL data. Type 'DELETE' to confirm: ")
            if confirm == "DELETE":
                try:
                    # Remove database file
                    db_path = Path("twitter_analyzer.db")
                    if db_path.exists():
                        db_path.unlink()
                        print("✅ Database cleared successfully!")
                        # Reinitialize
                        TwitterDatabase("twitter_analyzer.db")
                        print("🔄 New empty database created")
                    else:
                        print("No database file found")
                except Exception as e:
                    print(f"❌ Error clearing database: {e}")
            else:
                print("❌ Deletion cancelled")
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "stats":
            view_database_contents()
        elif command == "top":
            show_top_performers()
        else:
            print("Usage: python db_viewer.py [stats|top]")
    else:
        interactive_menu() 