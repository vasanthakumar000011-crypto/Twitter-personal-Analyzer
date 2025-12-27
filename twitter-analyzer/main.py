from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import json
import re
from datetime import datetime
import os
from pathlib import Path
import uvicorn
from dotenv import load_dotenv
import hashlib
import uuid

# Load environment variables from .env file
load_dotenv()

# Import database module
from database import db

# For AI processing - supports both OpenAI and local models
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import requests
    LOCAL_AI_AVAILABLE = True
except ImportError:
    LOCAL_AI_AVAILABLE = False

app = FastAPI(title="Twitter Persona Analyzer & Generator", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    """Serve the frontend"""
    return FileResponse('static/index.html')

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LOCAL_AI_URL = os.getenv("LOCAL_AI_URL", "http://localhost:11434")  # Ollama default
AI_MODEL = os.getenv("AI_MODEL", "gpt-3.5-turbo")  # or "llama2", "mistral", etc.
USE_LOCAL_AI = os.getenv("USE_LOCAL_AI", "false").lower() == "true"

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Generate dataset ID for uploads
def generate_dataset_id() -> str:
    """Generate a unique dataset ID"""
    return str(uuid.uuid4())[:8]

# Pydantic models
class TweetRequest(BaseModel):
    context: str
    persona_style: Optional[str] = "high_engagement"
    count: Optional[int] = 3
    tone: Optional[str] = "auto"

class PersonaAnalysis(BaseModel):
    writing_style: Dict[str, Any]
    content_themes: List[str]
    engagement_patterns: Dict[str, Any]
    successful_patterns: List[str]

class GeneratedTweet(BaseModel):
    content: str
    predicted_engagement: str
    style_similarity: str
    reasoning: str

class TweetResponse(BaseModel):
    tweets: List[GeneratedTweet]
    persona_used: str
    generation_timestamp: str

class AIGeneratedTweets(BaseModel):
    """Structured model for AI-generated tweets"""
    tweets: List[GeneratedTweet]

# AI Client setup
def get_ai_client():
    if USE_LOCAL_AI and LOCAL_AI_AVAILABLE:
        return "local"
    elif OPENAI_AVAILABLE and OPENAI_API_KEY:
        return OpenAI(api_key=OPENAI_API_KEY)
    else:
        raise Exception("No AI client available. Set OPENAI_API_KEY or LOCAL_AI_URL")

def call_ai_model(prompt: str, system_prompt: str = "", structured_output: bool = False) -> str:
    """Universal AI calling function for both OpenAI and local models"""
    try:
        if USE_LOCAL_AI:
            # Local AI call using OpenAI-compatible chat completions API
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            if structured_output:
                messages.append({
                    "role": "system", 
                    "content": "IMPORTANT: Respond with valid JSON only. No additional text or formatting."
                })
            
            response = requests.post(
                f"{LOCAL_AI_URL}/v1/chat/completions",
                json={
                    "model": AI_MODEL,
                    "messages": messages,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 1000,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code != 200:
                raise Exception(f"Local AI API error: {response.status_code}")
                
            return response.json()["choices"][0]["message"]["content"]
        else:
            # OpenAI call
            client = get_ai_client()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Use structured output for OpenAI if available
            if structured_output and "gpt-4" in AI_MODEL.lower():
                response = client.chat.completions.create(
                    model=AI_MODEL,
                    messages=messages,
                    temperature=0.7,
                    response_format={"type": "json_object"}
                )
            else:
                response = client.chat.completions.create(
                    model=AI_MODEL,
                    messages=messages,
                    temperature=0.7
                )
            
            return response.choices[0].message.content
            
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="AI API timeout")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"AI API Request Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI API Error: {str(e)}")

def call_ai_structured(prompt: str, system_prompt: str = "", response_model=None) -> Dict[str, Any]:
    """AI calling function with structured output parsing"""
    try:
        if USE_LOCAL_AI:
            # For local AI, use regular call and parse manually
            raw_response = call_ai_model(prompt, system_prompt)
            return parse_ai_response_manually(raw_response, response_model)
        else:
            # For OpenAI, try to use structured output if available
            client = get_ai_client()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Try OpenAI's structured output if model supports it
            if AI_MODEL in ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]:
                try:
                    response = client.chat.completions.create(
                        model=AI_MODEL,
                        messages=messages,
                        temperature=0.7,
                        response_format={"type": "json_object"}
                    )
                    raw_response = response.choices[0].message.content
                    return json.loads(raw_response)
                except:
                    # Fallback to regular call
                    pass
            
            # Regular OpenAI call with manual parsing
            response = client.chat.completions.create(
                model=AI_MODEL,
                messages=messages,
                temperature=0.7
            )
            raw_response = response.choices[0].message.content
            return parse_ai_response_manually(raw_response, response_model)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI API Error: {str(e)}")

def parse_ai_response_manually(raw_response: str, response_model=None) -> Dict[str, Any]:
    """Parse AI response manually with fallback strategies"""
    try:
        # First, try to parse as direct JSON
        if raw_response.strip().startswith('[') or raw_response.strip().startswith('{'):
            return json.loads(raw_response)
        
        # Look for JSON code blocks
        import re
        json_pattern = r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```'
        matches = re.findall(json_pattern, raw_response, re.DOTALL)
        if matches:
            return json.loads(matches[0])
        
        # Look for JSON without code blocks
        json_pattern = r'(\{.*?\}|\[.*?\])'
        matches = re.findall(json_pattern, raw_response, re.DOTALL)
        if matches:
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, (dict, list)):
                        return parsed
                except:
                    continue
        
        # Fallback: try to extract tweets from text
        return extract_tweets_from_text(raw_response)
        
    except Exception as e:
        print(f"Manual parsing failed: {e}")
        return extract_tweets_from_text(raw_response)

def extract_tweets_from_text(text: str) -> Dict[str, Any]:
    """Extract tweets from unstructured text as fallback"""
    print(f"DEBUG: Fallback text extraction called with: {text[:100]}...")
    
    lines = text.split('\n')
    tweets = []
    
    # Skip markdown/code block markers
    skip_patterns = ['```', '```json', '[', '{', ']', '}']
    
    for line in lines:
        line = line.strip()
        if not line or line in skip_patterns:
            continue
            
        # Look for actual tweet content (avoid JSON fragments)
        if (line.startswith('"content":') or 
            (len(line) > 20 and len(line) < 280 and 
             not line.startswith('{') and 
             not line.startswith('[') and
             not line.startswith('"predicted_engagement') and
             not line.startswith('"style_similarity') and
             not line.startswith('"reasoning'))):
            
            # Clean the content
            content = line
            if line.startswith('"content":'):
                # Extract from JSON field
                content = line.split(':', 1)[1].strip(' "')
            elif line.startswith('"') and line.endswith('"'):
                content = line.strip('"')
            
            content = content.strip()
            if len(content) > 10 and len(content) < 280:
                tweets.append({
                    "content": content,
                    "predicted_engagement": "Medium",
                    "style_similarity": "7",
                    "reasoning": "Extracted from AI response"
                })
    
    # If no tweets extracted, create a sensible fallback
    if not tweets:
        print("DEBUG: No tweets extracted, creating fallback")
        tweets = [{
            "content": "Just shipped something new! Excited to share it with the community 🚀 #buildinpublic",
            "predicted_engagement": "Medium", 
            "style_similarity": "6",
            "reasoning": "Fallback tweet - AI parsing failed"
        }]
    
    print(f"DEBUG: Extracted {len(tweets)} tweets")
    return {"tweets": tweets}

class TwitterAnalyzer:
    def __init__(self, df: pd.DataFrame):
        print(f"DEBUG: Initializing TwitterAnalyzer with {len(df)} tweets")
        
        # Check for required columns
        required_cols = ['content', 'likes', 'retweets', 'replies']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for analysis: {missing_cols}")
        
        # Clean data before processing
        self.df = df.copy()
        
        # Ensure numeric columns are actually numeric
        numeric_cols = ['likes', 'retweets', 'replies', 'quotes', 'bookmarks', 'views']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
        
        # Calculate engagement score
        try:
            self.df['engagement_score'] = self._calculate_engagement_score()
            print(f"DEBUG: Engagement scores calculated. Range: {self.df['engagement_score'].min():.2f} to {self.df['engagement_score'].max():.2f}")
        except Exception as e:
            print(f"DEBUG: Error calculating engagement scores: {e}")
            # Fallback to simple calculation
            self.df['engagement_score'] = self.df['likes'] + self.df['retweets'] + self.df['replies']
        
        # Calculate content length
        self.df['content_length'] = self.df['content'].astype(str).str.len()
        print(f"DEBUG: TwitterAnalyzer initialized successfully")
    
    def _calculate_engagement_score(self) -> pd.Series:
        """Calculate engagement score based on likes, retweets, replies, etc."""
        weights = {
            'likes': 1.0,
            'retweets': 2.0,
            'replies': 1.5,
            'quotes': 2.5,
            'bookmarks': 3.0
        }
        
        score = pd.Series(0, index=self.df.index)
        for metric, weight in weights.items():
            if metric in self.df.columns:
                score += self.df[metric].fillna(0) * weight
                
        # Normalize by views if available
        if 'views' in self.df.columns and self.df['views'].sum() > 0:
            score = (score / self.df['views'].fillna(1)) * 1000
            
        return score
    
    def get_top_performers(self, top_n: int = 20) -> pd.DataFrame:
        """Get top performing tweets"""
        return self.df.nlargest(top_n, 'engagement_score')
    
    def analyze_writing_patterns(self) -> Dict[str, Any]:
        """Analyze writing patterns from top performing tweets"""
        top_tweets = self.get_top_performers()
        
        patterns = {
            'avg_length': int(top_tweets['content_length'].mean()),
            'common_starters': self._get_common_starters(top_tweets['content']),
            'punctuation_usage': self._analyze_punctuation(top_tweets['content']),
            'emoji_usage': self._analyze_emojis(top_tweets['content']),
            'hashtag_patterns': self._analyze_hashtags(top_tweets['content']),
            'tone_indicators': self._analyze_tone(top_tweets['content'])
        }
        
        return patterns
    
    def _get_common_starters(self, contents: pd.Series) -> List[str]:
        """Find common ways tweets start"""
        starters = []
        for content in contents:
            words = content.split()[:3]
            if len(words) >= 2:
                starters.append(' '.join(words))
        
        from collections import Counter
        return [starter for starter, count in Counter(starters).most_common(5)]
    
    def _analyze_punctuation(self, contents: pd.Series) -> Dict[str, float]:
        """Analyze punctuation usage patterns"""
        total_tweets = len(contents)
        patterns = {
            'exclamation_rate': sum('!' in content for content in contents) / total_tweets,
            'question_rate': sum('?' in content for content in contents) / total_tweets,
            'caps_rate': sum(any(c.isupper() for c in content) for content in contents) / total_tweets,
            'ellipsis_rate': sum('...' in content for content in contents) / total_tweets
        }
        return patterns
    
    def _analyze_emojis(self, contents: pd.Series) -> Dict[str, Any]:
        """Analyze emoji usage"""
        emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+')
        
        emoji_tweets = sum(1 for content in contents if emoji_pattern.search(content))
        total_tweets = len(contents)
        
        return {
            'usage_rate': emoji_tweets / total_tweets,
            'avg_per_tweet': sum(len(emoji_pattern.findall(content)) for content in contents) / total_tweets
        }
    
    def _analyze_hashtags(self, contents: pd.Series) -> Dict[str, Any]:
        """Analyze hashtag patterns"""
        hashtag_pattern = re.compile(r'#\w+')
        
        hashtag_tweets = sum(1 for content in contents if hashtag_pattern.search(content))
        total_tweets = len(contents)
        all_hashtags = []
        
        for content in contents:
            all_hashtags.extend(hashtag_pattern.findall(content))
        
        from collections import Counter
        top_hashtags = Counter(all_hashtags).most_common(5)
        
        return {
            'usage_rate': hashtag_tweets / total_tweets,
            'avg_per_tweet': len(all_hashtags) / total_tweets,
            'popular_hashtags': [tag for tag, count in top_hashtags]
        }
    
    def _analyze_tone(self, contents: pd.Series) -> List[str]:
        """Identify tone indicators"""
        indicators = []
        
        # Simple tone analysis based on keywords
        positive_words = ['amazing', 'awesome', 'great', 'love', 'excited', 'happy', 'best']
        question_starters = ['how', 'what', 'why', 'when', 'where', 'which']
        action_words = ['build', 'create', 'launch', 'ship', 'made', 'working']
        
        for content in contents:
            content_lower = content.lower()
            if any(word in content_lower for word in positive_words):
                indicators.append('positive')
            if any(word in content_lower for word in question_starters):
                indicators.append('inquisitive')
            if any(word in content_lower for word in action_words):
                indicators.append('action-oriented')
        
        from collections import Counter
        return [tone for tone, count in Counter(indicators).most_common(3)]

# API Endpoints
@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload and process Twitter data CSV"""
    try:
        print(f"DEBUG: Processing file: {file.filename}")
        print(f"DEBUG: File content type: {file.content_type}")
        
        # Read CSV
        content = await file.read()
        print(f"DEBUG: File size: {len(content)} bytes")
        
        # Decode content
        try:
            decoded_content = content.decode('utf-8')
            print("DEBUG: File decoded successfully")
        except UnicodeDecodeError as e:
            print(f"DEBUG: UTF-8 decoding failed: {e}")
            # Try with different encoding
            try:
                decoded_content = content.decode('latin-1')
                print("DEBUG: File decoded with latin-1")
            except Exception as e2:
                raise HTTPException(status_code=400, detail=f"Unable to decode file: {str(e2)}")
        
        # Parse CSV
        try:
            df = pd.read_csv(pd.io.common.StringIO(decoded_content))
            print(f"DEBUG: CSV parsed successfully. Shape: {df.shape}")
            print(f"DEBUG: Columns: {list(df.columns)}")
        except pd.errors.EmptyDataError:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        except pd.errors.ParserError as e:
            raise HTTPException(status_code=400, detail=f"CSV parsing error: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")
        
        # Validate required columns
        required_cols = ['content', 'user', 'likes', 'retweets', 'replies']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            available_cols = list(df.columns)
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_cols}. Available columns: {available_cols}"
            )
        
        print("DEBUG: Column validation passed")
        
        # Check for empty dataframe
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV contains no data rows")
        
        # Generate unique dataset ID
        dataset_id = generate_dataset_id()
        print(f"DEBUG: Generated dataset ID: {dataset_id}")
        
        # Analyze the data
        try:
            analyzer = TwitterAnalyzer(df)
            print("DEBUG: TwitterAnalyzer created successfully")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating analyzer: {str(e)}")
        
        # Store tweets in database
        try:
            stored_count = db.store_tweets(df, dataset_id)
            print(f"DEBUG: Stored {stored_count} tweets in database")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error storing tweets in database: {str(e)}")
        
        # Get writing patterns
        try:
            writing_patterns = analyzer.analyze_writing_patterns()
            print("DEBUG: Writing patterns analyzed successfully")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error analyzing writing patterns: {str(e)}")
        
        # Get top performers for content analysis
        try:
            top_performers = analyzer.get_top_performers(20)
            print(f"DEBUG: Got {len(top_performers)} top performers")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting top performers: {str(e)}")
        
        # Create persona analysis using AI
        try:
            top_tweets_text = "\n---\n".join(top_performers['content'].tolist()[:10])
            print("DEBUG: Prepared top tweets for AI analysis")
            
            system_prompt = """You are an expert social media analyst. Analyze the provided tweets and extract:
                1. Writing style characteristics
                2. Content themes and topics
                3. Engagement patterns
                4. What makes these tweets successful
                Return your analysis in structured JSON format."""
            
            analysis_prompt = f"""
            Analyze these high-engagement tweets from the Build in Public community:
            
            {top_tweets_text}
            
            Statistical patterns found:
            - Average length: {writing_patterns['avg_length']} characters
            - Common starters: {writing_patterns['common_starters']}
            - Emoji usage: {writing_patterns['emoji_usage']['usage_rate']:.2%}
            - Hashtag usage: {writing_patterns['hashtag_patterns']['usage_rate']:.2%}
            
            Provide a comprehensive analysis of what makes these tweets successful.
            """
            
            print("DEBUG: About to call AI model")
            ai_analysis = call_ai_model(analysis_prompt, system_prompt)
            print("DEBUG: AI analysis completed successfully")
            
        except Exception as e:
            print(f"DEBUG: AI analysis failed: {str(e)}")
            # Continue without AI analysis for now
            ai_analysis = "AI analysis temporarily unavailable"
        
        # Store persona analysis in database
        try:
            persona_id = db.store_persona(
                name="build_in_public",
                dataset_id=dataset_id,
                total_tweets=len(df),
                writing_patterns=writing_patterns,
                ai_analysis=ai_analysis
            )
            print(f"DEBUG: Stored persona analysis with ID {persona_id}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error storing persona analysis: {str(e)}")
        
        return {
            "message": "CSV processed successfully",
            "dataset_id": dataset_id,
            "total_tweets": len(df),
            "top_performers_count": len(top_performers),
            "writing_patterns": writing_patterns,
            "ai_analysis": ai_analysis,
            "persona_id": persona_id
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as they are
        raise
    except Exception as e:
        # Catch any other unexpected errors
        error_msg = f"Unexpected error: {str(e)} | Type: {type(e).__name__}"
        print(f"DEBUG: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/generate-tweets", response_model=TweetResponse)
async def generate_tweets(request: TweetRequest):
    """Generate tweets based on analyzed persona and user context"""
    try:
        # Get persona data from database
        persona_data = db.get_persona("build_in_public")
        if not persona_data:
            raise HTTPException(
                status_code=400, 
                detail="No persona data available. Please upload CSV first."
            )
        
        # Get successful tweet examples from database
        top_performers_df = db.get_top_performers(persona_data['dataset_id'], 5)
        top_tweets = top_performers_df['content'].tolist()
        writing_patterns = persona_data['writing_patterns']
        
        # Create structured generation prompt
        system_prompt = """You are an expert Twitter content creator specializing in the Build in Public community style. 
        You MUST respond with valid JSON only. Do not include any text outside the JSON response.
        Generate engaging tweets that match the successful patterns while incorporating the user's context.
        Each tweet should be under 280 characters and follow the community's writing style.

        REQUIRED JSON format:
        {
        "tweets": [
            {
            "content": "tweet text here",
            "predicted_engagement": "Low/Medium/High", 
            "style_similarity": "1-10",
            "reasoning": "brief explanation"
            }
        ]
        }"""
        
        generation_prompt = f"""
        Generate {request.count} tweets about: {request.context}
        
        Style to match - Build in Public community examples:
        {chr(10).join([f"• {tweet}" for tweet in top_tweets])}
        
        Writing patterns to follow:
        - Average length: {writing_patterns.get('avg_length', 150)} characters
        - Emoji usage rate: {writing_patterns.get('emoji_usage', {}).get('usage_rate', 0.5):.1%}
        - Hashtag usage rate: {writing_patterns.get('hashtag_patterns', {}).get('usage_rate', 0.4):.1%}
        - Common starters: {', '.join(writing_patterns.get('common_starters', [])[:3])}
        - Popular hashtags: {', '.join(writing_patterns.get('hashtag_patterns', {}).get('popular_hashtags', [])[:3])}
        
        Requirements:
        1. Match the Build in Public community style exactly
        2. Incorporate the context naturally and authentically
        3. Under 280 characters each
        4. Authentic and engaging tone
        5. Include relevant emojis and hashtags when appropriate
        6. Use successful patterns from the examples
        
        Return ONLY valid JSON with the exact format specified above. No additional text.
        """
        
        ai_response = call_ai_model(generation_prompt, system_prompt, structured_output=True)
        
        # Parse AI response with robust error handling
        tweets_data = []
        try:
            # Clean the response - remove any non-JSON content
            cleaned_response = ai_response.strip()
            
            # Extract JSON if wrapped in markdown or other text
            if '```json' in cleaned_response:
                start = cleaned_response.find('```json') + 7
                end = cleaned_response.find('```', start)
                cleaned_response = cleaned_response[start:end].strip()
            elif '```' in cleaned_response:
                start = cleaned_response.find('```') + 3
                end = cleaned_response.rfind('```')
                cleaned_response = cleaned_response[start:end].strip()
            
            # Try to find JSON object in the response
            if '{' in cleaned_response and '}' in cleaned_response:
                start = cleaned_response.find('{')
                end = cleaned_response.rfind('}') + 1
                cleaned_response = cleaned_response[start:end]
            
            parsed_response = json.loads(cleaned_response)
            
            if "tweets" in parsed_response and isinstance(parsed_response["tweets"], list):
                tweets_data = parsed_response["tweets"][:request.count]
            else:
                raise ValueError("Invalid response structure")
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"JSON parsing failed: {e}")
            print(f"AI Response: {ai_response[:500]}...")
            
            # Robust fallback parsing
            tweets_data = []
            lines = [line.strip() for line in ai_response.split('\n') if line.strip()]
            
            current_tweet = {}
            for line in lines:
                # Look for content patterns
                if ('"content"' in line or line.startswith('"') or 
                    any(starter in line.lower() for starter in ['just', 'working', 'building', 'launched'])):
                    
                    # Extract content from various formats
                    content = line
                    if '"content"' in line:
                        content = line.split('"content"')[1].split('"')[1] if '"' in line.split('"content"')[1] else line
                    elif line.startswith('"') and line.endswith('"'):
                        content = line[1:-1]
                    elif line.startswith('- '):
                        content = line[2:]
                    
                    # Clean and validate content
                    content = content.strip().strip('"').strip("'").strip()
                    if 10 < len(content) < 280 and not content.lower().startswith('predicted'):
                        tweets_data.append({
                            "content": content,
                            "predicted_engagement": "Medium",
                            "style_similarity": "7",
                            "reasoning": "Generated using Build in Public patterns"
                        })
                        
                        if len(tweets_data) >= request.count:
                            break
            
            # Final fallback if no tweets parsed
            if not tweets_data:
                context_preview = request.context[:100] + "..." if len(request.context) > 100 else request.context
                tweets_data = [{
                    "content": f"Just working on {context_preview} 🚀 Excited to share the progress! #buildinpublic",
                    "predicted_engagement": "Medium", 
                    "style_similarity": "7",
                    "reasoning": "Fallback tweet using Build in Public style"
                }]
        
        # Validate and clean tweet data
        validated_tweets = []
        for i, tweet_data in enumerate(tweets_data[:request.count]):
            if isinstance(tweet_data, dict):
                # Ensure all required fields exist and are valid
                content = str(tweet_data.get("content", "")).strip()
                
                # Skip empty or invalid content
                if len(content) < 10:
                    continue
                    
                # Truncate if too long
                if len(content) > 280:
                    content = content[:277] + "..."
                
                validated_tweet = {
                    "content": content,
                    "predicted_engagement": tweet_data.get("predicted_engagement", "Medium"),
                    "style_similarity": str(tweet_data.get("style_similarity", "7")),
                    "reasoning": tweet_data.get("reasoning", "Generated using Build in Public patterns")
                }
                
                validated_tweets.append(validated_tweet)
            elif isinstance(tweet_data, str) and len(tweet_data) > 10:
                # Handle case where AI returns array of strings
                validated_tweets.append({
                    "content": tweet_data[:280],
                    "predicted_engagement": "Medium",
                    "style_similarity": "7", 
                    "reasoning": "Generated using Build in Public patterns"
                })
        
        # Ensure we have at least one tweet
        if not validated_tweets:
            context_preview = request.context[:80] + "..." if len(request.context) > 80 else request.context
            validated_tweets = [{
                "content": f"Just shipped something new! {context_preview} 🎉 The journey continues... #buildinpublic",
                "predicted_engagement": "Medium",
                "style_similarity": "8",
                "reasoning": "Fallback using successful Build in Public patterns"
            }]
        
        # Pad if we need more tweets than generated
        while len(validated_tweets) < request.count:
            base_tweet = validated_tweets[0] if validated_tweets else {
                "content": f"Working on {request.context[:50]}... exciting updates coming! 🚀",
                "predicted_engagement": "Medium", 
                "style_similarity": "7",
                "reasoning": "Generated variation"
            }
            
            # Create variations
            variations = [
                f"Progress update: {request.context[:60]}... loving the journey! ✨ #buildinpublic",
                f"Building in public: {request.context[:70]}... one step at a time 💪",
                f"Exciting news! Just made progress on {request.context[:50]}... 🎯 #indie"
            ]
            
            variation_idx = (len(validated_tweets) - 1) % len(variations)
            validated_tweets.append({
                "content": variations[variation_idx][:280],
                "predicted_engagement": "Medium",
                "style_similarity": "7", 
                "reasoning": f"Generated variation {variation_idx + 1}"
            })
        
        generated_tweets = [
            GeneratedTweet(**tweet) for tweet in validated_tweets[:request.count]
        ]
        
        # Store generated tweets in database
        try:
            for tweet in validated_tweets[:request.count]:
                db.store_generated_tweet(
                    persona_id=persona_data['id'],
                    context=request.context,
                    content=tweet['content'],
                    predicted_engagement=tweet['predicted_engagement'],
                    style_similarity=tweet['style_similarity'],
                    reasoning=tweet['reasoning']
                )
        except Exception as e:
            print(f"Warning: Failed to store generated tweets: {e}")
        
        return TweetResponse(
            tweets=generated_tweets,
            persona_used="build_in_public",
            generation_timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        print(f"Error in generate_tweets: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating tweets: {str(e)}")

@app.get("/persona-analysis")
async def get_persona_analysis():
    """Get the current persona analysis"""
    persona_data = db.get_persona("build_in_public")
    if not persona_data:
        raise HTTPException(
            status_code=400, 
            detail="No persona data available. Please upload CSV first."
        )
    
    # Get top performers sample from database
    top_performers_df = db.get_top_performers(persona_data['dataset_id'], 3)
    top_performers_sample = top_performers_df['content'].tolist()
    
    return {
        "total_tweets_analyzed": persona_data['total_tweets'],
        "writing_patterns": persona_data['writing_patterns'],
        "ai_analysis": persona_data['ai_analysis'],
        "top_performers_sample": top_performers_sample,
        "dataset_id": persona_data['dataset_id'],
        "created_at": persona_data['created_at'],
        "updated_at": persona_data['updated_at']
    }

@app.get("/test-ai")
async def test_ai():
    """Test AI configuration"""
    try:
        # Check environment variables
        env_status = {
            "USE_LOCAL_AI": USE_LOCAL_AI,
            "AI_MODEL": AI_MODEL,
            "LOCAL_AI_URL": LOCAL_AI_URL,
            "OPENAI_API_KEY_SET": bool(OPENAI_API_KEY),
            "OPENAI_AVAILABLE": OPENAI_AVAILABLE,
            "LOCAL_AI_AVAILABLE": LOCAL_AI_AVAILABLE
        }
        
        # Try to get AI client
        try:
            client = get_ai_client()
            client_status = "OK" if client else "Failed"
        except Exception as e:
            client_status = f"Error: {str(e)}"
        
        # Try a simple AI call
        try:
            test_response = call_ai_model("Say 'Hello, I am working!' in exactly those words.", "You are a helpful assistant.")
            ai_call_status = "OK" if test_response else "No response"
        except Exception as e:
            ai_call_status = f"Error: {str(e)}"
            test_response = None
        
        # Test structured output
        try:
            structured_test = call_ai_structured(
                'Generate 1 test tweet about "building in public" in JSON format.',
                'You must respond with valid JSON only. Format: {"tweets": [{"content": "tweet text", "predicted_engagement": "Medium", "style_similarity": "7", "reasoning": "test"}]}'
            )
            structured_status = "OK" if "tweets" in structured_test else "Failed"
        except Exception as e:
            structured_status = f"Error: {str(e)}"
            structured_test = None
        
        return {
            "environment": env_status,
            "client_status": client_status,
            "ai_call_status": ai_call_status,
            "test_response": test_response,
            "structured_status": structured_status,
            "structured_test": structured_test
        }
        
    except Exception as e:
        return {
            "error": f"Test failed: {str(e)}",
            "type": type(e).__name__
        }

@app.get("/database-stats")
async def get_database_stats():
    """Get database statistics"""
    try:
        return db.get_database_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting database stats: {str(e)}")

@app.get("/personas")
async def list_personas():
    """List all available personas"""
    try:
        stats = db.get_database_stats()
        # For now, we'll get the build_in_public persona if it exists
        persona = db.get_persona("build_in_public")
        personas = [persona] if persona else []
        
        return {
            "total_personas": stats["total_personas"],
            "personas": personas
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing personas: {str(e)}")

@app.get("/generated-tweets/{persona_name}")
async def get_generated_tweets_history(persona_name: str, limit: int = 50):
    """Get history of generated tweets for a persona"""
    try:
        persona_data = db.get_persona(persona_name)
        if not persona_data:
            raise HTTPException(status_code=404, detail=f"Persona '{persona_name}' not found")
        
        tweets = db.get_generated_tweets(persona_data['id'], limit)
        return {
            "persona_name": persona_name,
            "total_tweets": len(tweets),
            "tweets": tweets
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting generated tweets: {str(e)}")

@app.delete("/dataset/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a complete dataset and associated data"""
    try:
        db.clear_dataset(dataset_id)
        return {"message": f"Dataset {dataset_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting dataset: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    ai_status = "OpenAI" if not USE_LOCAL_AI else f"Local AI ({LOCAL_AI_URL})"
    
    try:
        db_stats = db.get_database_stats()
    except Exception as e:
        db_stats = {"error": str(e)}
    
    return {
        "status": "healthy",
        "ai_backend": ai_status,
        "model": AI_MODEL,
        "database_stats": db_stats
    }

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, reload=DEBUG)