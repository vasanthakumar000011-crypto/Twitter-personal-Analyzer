import json
import csv
import glob
import os

def extract_tweets_from_file(json_file):
    """Extract tweets from a single JSON file (supports both Community and User timelines)"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    tweets = []
    
    # Detect timeline type and get the appropriate timeline object
    timeline = None
    timeline_type = "Unknown"
    
    # Check for Community timeline
    community_result = data.get('data', {}).get('communityResults', {}).get('result', {})
    if community_result.get('__typename') == 'Community':
        timeline = community_result.get('ranked_community_timeline', {}).get('timeline', {})
        timeline_type = "Community"
    
    # Check for User timeline
    user_result = data.get('data', {}).get('user', {}).get('result', {})
    if user_result.get('__typename') == 'User':
        timeline = user_result.get('timeline', {}).get('timeline', {})
        timeline_type = "User"
    
    if not timeline:
        print(f"  Warning: Unknown timeline format in {os.path.basename(json_file)}")
        return []
    
    print(f"  Detected: {timeline_type} timeline")
    
    instructions = timeline.get('instructions', [])
    
    for instruction in instructions:
        if instruction.get('type') == 'TimelineAddEntries':
            entries = instruction.get('entries', [])
            
            for entry in entries:
                # Check if it's a timeline item with tweet content
                content = entry.get('content', {})
                if content.get('entryType') == 'TimelineTimelineItem':
                    item_content = content.get('itemContent', {})
                    if item_content.get('itemType') == 'TimelineTweet':
                        
                        tweet_result = item_content['tweet_results']['result']
                        
                        # Extract tweet data
                        if 'legacy' in tweet_result:
                            legacy = tweet_result['legacy']
                            user_core = tweet_result.get('core', {}).get('user_results', {}).get('result', {}).get('core', {})
                            
                            # Extract views count
                            views_count = 0
                            views_data = tweet_result.get('views', {})
                            if views_data and views_data.get('state') == 'EnabledWithCount':
                                try:
                                    views_count = int(views_data.get('count', '0').replace(',', ''))
                                except:
                                    views_count = 0
                            
                            tweet_info = {
                                'content': legacy.get('full_text', ''),
                                'user': user_core.get('screen_name', ''),
                                'user_name': user_core.get('name', ''),
                                'likes': legacy.get('favorite_count', 0),
                                'retweets': legacy.get('retweet_count', 0),
                                'replies': legacy.get('reply_count', 0),
                                'quotes': legacy.get('quote_count', 0),
                                'bookmarks': legacy.get('bookmark_count', 0),
                                'views': views_count,
                                'created_at': legacy.get('created_at', ''),
                                'tweet_id': legacy.get('id_str', ''),
                                'timeline_type': timeline_type,  # Track source timeline type
                                'source_file': os.path.basename(json_file)  # Add source file info
                            }
                            
                            if tweet_info['content']:  # Only add if there's actual content
                                tweets.append(tweet_info)
    
    return tweets

def extract_tweets_from_folder(folder_path):
    """Extract tweets from all JSON files in a folder"""
    all_tweets = []
    
    # Find all JSON files matching the pattern
    json_pattern = os.path.join(folder_path, "*-*.json")
    json_files = glob.glob(json_pattern)
    
    # Also check for single files like profile-tweets.json
    single_files = glob.glob(os.path.join(folder_path, "*.json"))
    for file in single_files:
        if file not in json_files and not os.path.basename(file).startswith('.'):
            json_files.append(file)
    
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return []
    
    json_files.sort()  # Sort files for consistent processing order
    
    print(f"Found {len(json_files)} JSON files to process:")
    for json_file in json_files:
        print(f"  - {os.path.basename(json_file)}")
    
    # Process each file
    for json_file in json_files:
        print(f"\nProcessing {os.path.basename(json_file)}...")
        try:
            tweets = extract_tweets_from_file(json_file)
            print(f"  Found {len(tweets)} tweets")
            all_tweets.extend(tweets)
        except Exception as e:
            print(f"  Error processing {json_file}: {e}")
    
    return all_tweets

def save_to_csv(tweets, filename='all_extracted_tweets.csv'):
    if not tweets:
        print("No tweets found!")
        return
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=tweets[0].keys())
        writer.writeheader()
        writer.writerows(tweets)
    
    print(f"\nSaved {len(tweets)} total tweets to {filename}")

if __name__ == "__main__":
    # Check if folder path is provided as argument
    import sys
    
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = input("Enter the folder path containing JSON files (or press Enter for current directory): ").strip()
        if not folder_path:
            folder_path = "."
    
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist!")
        exit(1)
    
    print(f"Looking for JSON files in: {os.path.abspath(folder_path)}")
    
    # Extract tweets from all files
    all_tweets = extract_tweets_from_folder(folder_path)
    
    if all_tweets:
        # Print summary
        print(f"\n=== SUMMARY ===")
        print(f"Total tweets extracted: {len(all_tweets)}")
        
        # Show breakdown by source file and timeline type
        from collections import Counter
        file_counts = Counter(tweet['source_file'] for tweet in all_tweets)
        timeline_counts = Counter(tweet['timeline_type'] for tweet in all_tweets)
        
        print(f"\nTweets per file:")
        for file, count in sorted(file_counts.items()):
            print(f"  {file}: {count} tweets")
        
        print(f"\nTweets by timeline type:")
        for timeline_type, count in timeline_counts.items():
            print(f"  {timeline_type}: {count} tweets")
        
        # Print first few tweets to verify
        print(f"\n=== First 3 tweets ===")
        for i, tweet in enumerate(all_tweets[:3]):
            print(f"\nTweet {i+1} ({tweet['timeline_type']} from {tweet['source_file']}):")
            print(f"  User: @{tweet['user']}")
            print(f"  Content: {tweet['content'][:80]}...")
            print(f"  Engagement: {tweet['likes']} likes, {tweet['retweets']} retweets, {tweet['replies']} replies")
            print(f"  Reach: {tweet['views']:,} views")
        
        # Save to CSV
        save_to_csv(all_tweets)
    else:
        print("No tweets found in any files!") 