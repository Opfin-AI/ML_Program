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
    NEWS_API_KEY = 'your_newsapi_key_here'  # Get from https://newsapi.org/register
    NEWS_API_URL = 'https://newsapi.org/v2/everything'

    # OpenAI API
    OPENAI_KEY = 'your_openai_api_key_here'  # Get from https://platform.openai.com/signup


def fetch_finnhub_news_for_day(date, symbol) -> list:
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
    
    
def fetch_newsapi_news_for_day(date, query) -> list: 
    """Fetch general news using NewsAPI"""
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
            print(f"No articles found for {date} from Finnhub")
    else:
        for article in finnhub_articles:
            ts = datetime.fromtimestamp(article['datetime'], tz=zone)
            '''print(f"🗞️ {ts}: {article['headline']}")
            print(f"   → {article['summary']}")
            print(f"   → Source: {article['source']} | URL: {article['url']}")
            print('-' * 80)'''
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
    
# OPEN AI SENTIMENT ANALYSIS METHOD
def analyze_sentiment_scores(day_list, target):
    client = OpenAI(api_key=OPENAI_KEY)

    prompt = (
        f"You are a financial news sentiment evaluator.\n\n"
        f"You will receive a Python list of N news headlines.\n"
        f"For each headline, assign a sentiment score: 1 if it's positive for {target}, 0 otherwise.\n\n"
        f"Return a JSON object with one field: 'sentiment_scores', whose value is a list of exactly {len(day_list)} integers (0 or 1), "
        f"in the same order as the headlines.\n"
        f"Do not skip any headlines. Do not explain. Only output the JSON object.\n\n"
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
    '''
    # Extract just the scores from the JSON lines
    # Optional: clean up markdown formatting if it appears
    cleaned = raw_output.strip().removeprefix("
json").removesuffix("
")

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", e)
        print("Raw response:\n", raw_output)
        parsed = []
    #MANUALLY CALCULATING OVERALL_DAY_SCORE FOR SENTIMENT BCUZ LLM IS BRAINDEAD AND CANNOT RELIABLY COUNT
    for day, info in parsed.items():
        headlines = info['headlines']
        if not headlines:
            overall = None
        else:
            ones = sum(h['sentiment_score'] for h in headlines)
            zeros = len(headlines) - ones
            overall = 1 if ones > zeros else 0
        parsed[day]['overall_day_score'] = overall
    # Extract just the overall sentiment scores that I manually computed and assigned to each day in the returned json
    overall_day_scores = [parsed[d]["overall_day_score"] for d in sorted(parsed, key=lambda x: int(x))]'''
    sentiment_scores = json.loads(raw_output)
    # sent_reasons = [item['reasoning'] for item in parsed if 'reasoning' in item]
    return sentiment_scores
    
def get_sentiment_scores(target: str, ticker: str, days_back: int = 10) -> int:
    """Get sentiment score for a specific ticker over the last 10 days"""
    
    api_configuration() # Configure API keys and URLs

    ET = ZoneInfo('America/New_York')

    end_date = datetime.now(tz=ET).date()
    start_date = end_date - timedelta(days=10)

    all_headlines = []
    days_to_headlines = {}
    
    current = start_date
    day = 0
    
    # creates a list to append the overall day scores to scorelist
    scorelist = []

    # EXTRACTING ARTICLES BY DAY FROM FINNHUB AND NEWS API
    while current <= end_date:
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
        gennews_articles = fetch_newsapi_news_for_day(current, target)
                
        # params = {
        #     'symbol': ticker,
        #     'from': current.strftime('%Y-%m-%d'),
        #     'to': current.strftime('%Y-%m-%d'),
        #     'token': FINNHUB_API_KEY
        # }

        # response = requests.get(FINNHUB_URL, params=params)
        # try:
        #     data = response.json()
        # except Exception as e:
        #     print(f"Failed to parse response: {e}")
        #     current += timedelta(days=1)
        #     continue

        # if not isinstance(data, list):
        #     print(f"Unexpected response: {data}")
        #     current += timedelta(days=1)
        #     continue

        # Transform Finnhub articles into headlines and append to day_list
        finnhub_headlines = finnhub_articles_into_headlines(ET, current, finnhub_articles)
        # print(f"Finnhub headlines for {current}: {finnhub_headlines}")
        day_list.append(finnhub_headlines)
        
        # Transform general news articles into headlines and append to day_list
        newsapi_headlines = newsapi_articles_into_headlines(current, gennews_articles)
        # print(f"General News headlines for {current}: {gennews_headlines}")
        day_list.append(newsapi_headlines)
        
        # if not data:
        #     print(f"No articles found for {current}")
        # else:
        #     for article in data:
        #         ts = datetime.fromtimestamp(article['datetime'], tz=ET)
        #         '''print(f"🗞️ {ts}: {article['headline']}")
        #         print(f"   → {article['summary']}")
        #         print(f"   → Source: {article['source']} | URL: {article['url']}")
        #         print('-' * 80)'''
        #         day_list.append(article['headline'])
        #         # print(f"day_list: {day_list}")
        #         all_headlines.append(article['headline'])
        #         # print(f"all_headlines: {all_headlines}")
                
        # days_to_headlines[day] = day_list
        
        # Analyze sentiment scores for the collected headlines
        day_scores = analyze_sentiment_scores(day_list, target)
        #print(day_list)
        print(f"SENTIMENT SCORES FOR THE DAY: {day_scores['sentiment_scores']}")
        # If the sum of the sentiment scores is greater than half the number of scores, consider it a positive day
        overall_day_score = 1 if sum(day_scores['sentiment_scores']) > len(day_scores['sentiment_scores']) / 2 else 0
        scorelist.append(overall_day_score)
        print(f"OVERALL DAY SCORE: {overall_day_score}")
        # if len(day_scores['sentiment_scores']) == len(day_list):
        #     print("equal length")
        # elif len(day_scores['sentiment_scores']) > len(day_list):
        #     print("more scores then headlines")
        # elif len(day_scores['sentiment_scores']) < len(day_list):
        #     print("less scores then headlines")
        current += timedelta(days=1)
        print(f"SCORE LIST: {scorelist}")
        time.sleep(2)  # Increased delay since we're hitting two APIs

    return scorelist

def get_final_sentiment(target, ticker: str) -> int:
    """Calculate final sentiment based on the scorelist"""
    scorelist = get_sentiment_scores(target, ticker)
    
    # 1/3 Majority Logic
    # === FINAL SENTIMENT DECISION BASED ON SCORELIST ===
    positive_days = sum(scorelist)
    threshold = len(scorelist) / 3

    if positive_days >= threshold:
        final_sentiment = 1
    else:
        final_sentiment = 0

    print("\nSummary of Sentiment over 10 Days:")
    print(f"Positive Days: {positive_days} out of {len(scorelist)}")
    print(f"Threshold (1/3 of days): {threshold}")
    print(f"Final Sentiment Outlook for {target}: {'Positive' if final_sentiment == 1 else 'Negative/Neutral'}")
    #returns final sentiment
    return final_sentiment


# main code to test the logic
ticker = "GLD"
target = ("Gold")
sentiment = get_final_sentiment(target, ticker)
print(f"Final Sentiment for {ticker}: {'Positive' if sentiment == 1 else 'Negative/Neutral'}")