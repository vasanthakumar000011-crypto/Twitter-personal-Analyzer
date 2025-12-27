import json

def explore_json_structure(obj, path="", level=0, max_level=3):
    if level > max_level:
        return
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = f"{path}.{key}" if path else key
            print("  " * level + f"{key}: {type(value).__name__}")
            
            if key in ['instructions', 'entries', 'timeline']:
                print("  " * level + f"  -> Found {key}!")
                if isinstance(value, list) and len(value) > 0:
                    print("  " * level + f"     First item type: {type(value[0]).__name__}")
                    if isinstance(value[0], dict):
                        print("  " * level + f"     First item keys: {list(value[0].keys())}")
            
            explore_json_structure(value, current_path, level + 1, max_level)
    elif isinstance(obj, list) and len(obj) > 0:
        print("  " * level + f"List with {len(obj)} items, first item type: {type(obj[0]).__name__}")
        if len(obj) > 0:
            explore_json_structure(obj[0], f"{path}[0]", level + 1, max_level)

with open('bip-network-tweets.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=== JSON Structure ===")
explore_json_structure(data)

print("\n=== Looking for timeline specifically ===")
timeline = data.get('data', {}).get('communityResults', {}).get('result', {}).get('ranked_community_timeline', {}).get('timeline', {})
print(f"Timeline found: {bool(timeline)}")
if timeline:
    print(f"Timeline keys: {list(timeline.keys())}")
    
instructions = timeline.get('instructions', [])
print(f"Instructions found: {len(instructions)} items")
for i, instruction in enumerate(instructions):
    print(f"Instruction {i}: type={instruction.get('type')}, keys={list(instruction.keys())}")
    
    if instruction.get('type') == 'TimelineAddEntries':
        entries = instruction.get('entries', [])
        print(f"  Found {len(entries)} entries")
        
        for j, entry in enumerate(entries[:3]):  # Look at first 3 entries
            print(f"  Entry {j}: keys={list(entry.keys())}")
            print(f"    entryType={entry.get('entryType')}")
            
            # Look at content regardless of entryType
            content = entry.get('content', {})
            if content:
                print(f"    Content keys: {list(content.keys())}")
                print(f"    Content entryType: {content.get('entryType')}")
                
                # Check itemContent
                item_content = content.get('itemContent', {})
                if item_content:
                    print(f"    ItemContent keys: {list(item_content.keys())}")
                    print(f"    ItemContent itemType: {item_content.get('itemType')}")
            
            if content.get('entryType') == 'TimelineTimelineItem':
                item_content = content.get('itemContent', {})
                
                if item_content.get('itemType') == 'TimelineTweet':
                    tweet_results = item_content.get('tweet_results', {})
                    result = tweet_results.get('result', {})
                    print(f"    Tweet result keys: {list(result.keys()) if result else 'None'}")
                    
                    if 'legacy' in result:
                        legacy = result['legacy']
                        print(f"    Legacy keys: {list(legacy.keys())}")
                        print(f"    Full text sample: {legacy.get('full_text', '')[:50]}...")
                        print(f"    Favorite count: {legacy.get('favorite_count', 0)}")
                        
                        # User info
                        print(f"    Core keys: {list(result.get('core', {}).keys())}")
                        user_results = result.get('core', {}).get('user_results', {})
                        print(f"    User results keys: {list(user_results.keys()) if user_results else 'None'}")
                        
                        if user_results:
                            user_result = user_results.get('result', {})
                            print(f"    User result keys: {list(user_result.keys()) if user_result else 'None'}")
                            
                            # Check user core
                            user_core = user_result.get('core', {})
                            if user_core:
                                print(f"    User core keys: {list(user_core.keys())}")
                                print(f"    User core screen_name: {user_core.get('screen_name', 'NO_SCREEN_NAME')}")
                                print(f"    User core name: {user_core.get('name', 'NO_NAME')}")
                            
                            if user_result and 'legacy' in user_result:
                                user_legacy = user_result['legacy']
                                print(f"    User legacy screen_name: {user_legacy.get('screen_name', 'NO_SCREEN_NAME')}")
                                print(f"    User legacy name: {user_legacy.get('name', 'NO_NAME')}")
                            else:
                                print(f"    No user legacy found") 