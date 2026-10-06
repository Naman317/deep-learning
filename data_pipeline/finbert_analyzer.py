import yfinance as yf

_finbert_pipeline = None


def load_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        try:
            from transformers import pipeline
            _finbert_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        except Exception:
            _finbert_pipeline = None
    return _finbert_pipeline


def analyze_news_sentiment(ticker, max_articles=5):
    nlp = load_finbert()
    articles = []
    total_score = 0.0
    
    try:
        raw_news = yf.Ticker(ticker).news
        if raw_news:
            for item in raw_news[:max_articles]:
                content = item.get('content', {})
                title = content.get('title', item.get('title', 'No Title'))
                url = content.get('clickThroughUrl', {}).get('url', item.get('link', '#'))
                
                label = "NEUTRAL"
                score = 0.0
                color = "#6B7280"
                
                if title and title != "No Title" and nlp is not None:
                    try:
                        res = nlp(title)[0]
                        label = res['label'].upper()
                        score = float(res['score'])
                        
                        if label == 'POSITIVE':
                            total_score += score
                            color = "#10B981"
                        elif label == 'NEGATIVE':
                            total_score -= score
                            color = "#EF4444"
                    except Exception:
                        pass
                        
                articles.append({
                    'title': title,
                    'url': url,
                    'label': label,
                    'score': score,
                    'color': color
                })
    except Exception:
        pass
        
    composite_score = total_score / len(articles) if articles else 0.0
    return composite_score, articles
