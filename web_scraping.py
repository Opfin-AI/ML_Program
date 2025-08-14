import time
from datetime import datetime, timedelta, time as dt_time
import requests
from zoneinfo import ZoneInfo
import json
from openai import OpenAI
from openai.types import ResponseFormatJSONObject
from openai.types.chat.completion_create_params import ResponseFormat


def api_configuration():
    """Configure API keys and URLs."""
    global FINNHUB_API_KEY, FINNHUB_URL, NEWS_API_KEY, NEWS_API_URL, OPENAI_KEY
    # Finnhub API (Financial News)
    FINNHUB_API_KEY = 'your_finnhub_api_key_here'  # Get from https://finnhub.io/dashboard
    FINNHUB_URL = 'https://finnhub.io/api/v1/company-news'
    # NewsAPI (General News)
    NEWS_API_KEY = 'your_news_api_key_here'  # Get from https://newsapi.org/register
    NEWS_API_URL = 'https://newsapi.org/v2/everything'
    # OpenAI API
    OPENAI_KEY = 'your_openai_api_key_here'  # Get from https://platform.openai.com/signup


def fetch_finnhub_news_for_day(date, ticker: str) -> list:
    """Fetch financial news from Finnhub"""
    params = {
        'symbol': ticker,
        'from': date.strftime('%Y-%m-%d'),
        'to': date.strftime('%Y-%m-%d'),
        'token': FINNHUB_API_KEY
    }
        
    try:
        response = requests.get(FINNHUB_URL, params=params)
        data = response.json()
        
        if isinstance(data, list):
            return data
        else:
            print(f"Unexpected response: {data}")
            date += timedelta(days=1)
            return []
    except Exception as e:
        print(f"Failed to parse response: {e}")
        date += timedelta(days=1)
        return []
    
    
def fetch_newsapi_news_for_day(date, company_name, ticker) -> list: 
    """Fetch general news using NewsAPI"""
    query = f'"{company_name}" OR "{ticker}"'
    params = {
        'q': query,
        'from': date.strftime('%Y-%m-%d'),
        'to': date.strftime('%Y-%m-%d'),
        'sortBy': 'relevancy',
        'language': 'en',
        'apiKey': NEWS_API_KEY,
        'pageSize': 50  # Limit to avoid too many results
    }
    
    try:
        response = requests.get(NEWS_API_URL, params=params)
        data = response.json()
        
        if data.get('status') == 'ok' and data.get('articles'):
            return data['articles']
        else:
            if 'message' in data:
                print(f"NewsAPI message: {data['message']}")
            return []
    except Exception as e:
        print(f"Error fetching general news: {e}")
        return []
    
    
def finnhub_articles_into_headlines(zone, date, finnhub_articles: list) -> list:
    """Convert Finnhub articles into headlines"""
    finnhub_headlines = []

    if not finnhub_articles:
        print(f"No articles found for {date} from Finnhub API")
    else:
        for article in finnhub_articles:
            ts = datetime.fromtimestamp(article['datetime'], tz=zone)
            finnhub_headlines.append(article['headline'])
    return finnhub_headlines
  
        
def newsapi_articles_into_headlines(date, newsapi_articles: list) -> list:
    """Convert NewsAPI articles into headlines"""
    newsapi_headlines = []

    if not newsapi_articles:
        print(f"No articles found for {date} from NewsAPI")
    else:
        for article in newsapi_articles:
            if article.get('title'):
                newsapi_headlines.append(article['title'])
    return newsapi_headlines
   
    
def analyze_sentiment_scores(day_list: list, target) -> dict:
    """Analyze sentiment scores using OpenAI API"""
    client = OpenAI(api_key=OPENAI_KEY)

    prompt = (
        "You are a financial news sentiment evaluator.\n\n"
        "You will receive a Python list of N news headlines.\n"
        f"For each headline, assign a sentiment score: 1 if it's positive for {target}, 0 otherwise.\n\n"
        f"Return a JSON object with one field: 'sentiment_scores', whose value is a list of exactly {len(day_list)} integers (0 or 1), "
        "in the same order as the headlines.\n"
        "Do not skip any headlines. Do not explain. Only output the JSON object.\n\n"
        f"headlines = {json.dumps(day_list)}"

    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
                "content": f"You analyze financial news and score headlines based on their impact on {target}."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0

    )

    # Parse each JSON object from the response text
    raw_output = response.choices[0].message.content
    sentiment_scores = json.loads(raw_output)
    return sentiment_scores
    
    
def get_sentiment_scores(target, ticker: str, days_back: int) -> int:
    """Get sentiment score for a specific ticker over the last 30 days"""
    
    api_configuration() # Configure API keys and URLs
    ET = ZoneInfo('America/New_York')
    end_date = datetime.now(tz=ET).date()
    start_date = end_date - timedelta(days_back)
    current = start_date
    day = 0
    
    # creates a list to append the overall day scores to scorelist
    scorelist = []
    # calculates proportion of positive sentiment scores for each day
    prop_pos_dict = {}
    
    # EXTRACTING ARTICLES BY DAY FOR 10 DAYS FROM FINNHUB AND NEWS API
    while current < end_date:
        day = day + 1
        day_list = []
        print("---------------------------------------------------------------------------")
        print("Day: " + str(day))
        print()
        print()
        
        # Fetch financial news from Finnhub
        print("Fetching financial news from Finnhub...")
        finnhub_articles = fetch_finnhub_news_for_day(current, ticker)
    
        # Fetch general news from NewsAPI
        print("Fetching general news from NewsAPI...")
        newsapi_articles = fetch_newsapi_news_for_day(current, target[0], target[1])

        # Transform Finnhub articles into headlines and append to day_list
        finnhub_headlines = finnhub_articles_into_headlines(ET, current, finnhub_articles)
            
        for headline in finnhub_headlines:
            if headline:  # Check if the headline is not empty
                day_list.append(headline)
                
        # Transform general news articles into headlines and append to day_list
        newsapi_headlines = newsapi_articles_into_headlines(current, newsapi_articles)
            
        for headline in newsapi_headlines:
            if headline:  # Check if the headline is not empty
                day_list.append(headline)
        print(f"Total headlines for {current}: {len(day_list)}\n")
        
        # if no headlines found for the day, continue with logic
        if(len(day_list) == 0):
            print(f"ALERT: No headlines found for {current}, overall day score = 0\n")
            overall_day_score = 0
            scorelist.append(overall_day_score)
            prop_pos_dict[day] = 0
            current += timedelta(days=1)
            time.sleep(2)  # Increased delay since we're hitting two APIs
            continue        
                
        # Analyze sentiment scores for the collected headlines
        day_scores = analyze_sentiment_scores(day_list, target)
        print(f"Sentiment scores for day: {day_scores['sentiment_scores']}\n")
        
        # 1/3 Majority Logic - for each sentiment in day_scores['sentiment_scores']
        positive_scores = sum(day_scores['sentiment_scores'])
        print(f"Total positive sentiment scores for {current}: {positive_scores}")
        threshold = len(day_scores['sentiment_scores']) / 3
        prop_pos_dict[day] = positive_scores / len(day_scores['sentiment_scores'])
        
        if positive_scores >= threshold:
            overall_day_score = 1
        else:
            overall_day_score = 0
        
        print(f"OVERALL DAY SCORE: {overall_day_score}")
        scorelist.append(overall_day_score)
        
        current += timedelta(days=1)
        print(f"SCORE LIST: {scorelist}\n")
        time.sleep(2)  # two second delay since we're hitting two APIs

    # alters scorelist to have minimum one postive day score for xgboost to work properly
    if sum(scorelist) == 0:
        print("ALERT: ALTERING SCORELIST - minimum one positive overall_day_score for xgboost to work\n")
        highest_prop_pos_day = max(prop_pos_dict, key=prop_pos_dict.get)
        scorelist[highest_prop_pos_day - 1] = 1
        print(f"ALTERED SCORELIST BASED ON XGBOOST CONDITIONS: {scorelist}\n")
            
    # Return the scorelist containing sentiment scores for each day
    return scorelist


def print_results(target, ticker: str, days_back: int):
    """Prints out information about scorelist"""
    scorelist = get_sentiment_scores(target, ticker, days_back)
    positive_days = sum(scorelist)

    print(f"\nSummary of Sentiments over {days_back} Days:")
    print(f"Positive Days: {positive_days} out of {len(scorelist)}")
    print(f"Final Score List for {target}: {scorelist}")


# main code to test the logic
ticker = "NFLX"
target = ("Netflix", ticker)
print_results(target, ticker, 30)