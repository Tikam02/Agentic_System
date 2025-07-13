import feedparser
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import re
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import ollama
from abc import ABC, abstractmethod
import os

def safe_json_dumps(obj, **kwargs):
    """Convert numpy/pandas types to JSON serializable types"""
    def convert_types(o):
        if isinstance(o, (np.integer, np.floating)):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif pd.isna(o):
            return None
        return o
    
    return json.dumps(obj, default=convert_types, **kwargs)

class BaseAgent(ABC):
    def __init__(self, name: str, llm_client):
        self.name = name
        self.llm = llm_client
        self.memory = []
    
    @abstractmethod
    def execute(self, input_data: Any) -> Dict:
        pass
    
    def _llm_call(self, prompt: str) -> str:
        try:
            response = self.llm.chat(
                model='llama3.2:latest',
                messages=[{'role': 'user', 'content': prompt}]
            )
            return response['message']['content']
        except Exception as e:
            return f"LLM Error: {e}"

class CSVDataAgent(BaseAgent):
    def __init__(self, llm_client, csv_path="./NSE_ALL.csv"):
        super().__init__("CSV_Data_Agent", llm_client)
        self.csv_path = csv_path
    
    def execute(self, input_data: Dict) -> Dict:
        """Read NSE stock data from CSV"""
        try:
            df = pd.read_csv(self.csv_path)
            
            # Extract symbols - they already have .NS suffix
            symbols = df['Symbol'].tolist()
            stock_list = symbols[:50]  # Limit for demo
            
            return {
                "agent": self.name,
                "stock_symbols": stock_list,
                "total_stocks": len(stock_list),
                "csv_data": df.to_dict('records')[:50]
            }
            
        except Exception as e:
            print(f"CSV read error: {e}")
            # Fallback to default list
            default_stocks = [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "HINDUNILVR.NS",
                "INFY.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "ASIANPAINT.NS"
            ]
            return {
                "agent": self.name,
                "stock_symbols": default_stocks,
                "total_stocks": len(default_stocks),
                "csv_data": []
            }

class SentimentAgent(BaseAgent):
    def __init__(self, llm_client):
        super().__init__("Sentiment_Agent", llm_client)
    
    def execute(self, input_data: Dict) -> Dict:
        """Analyze sentiment for each stock based on price and news"""
        price_data = input_data.get('price_data', {})
        news_data = input_data.get('news_data', {})
        
        sentiment_analysis = self._analyze_stock_sentiments(price_data, news_data)
        market_sentiment = self._calculate_market_sentiment(sentiment_analysis)
        
        return {
            "agent": self.name,
            "stock_sentiments": sentiment_analysis,
            "market_sentiment": market_sentiment,
            "bullish_count": len([s for s in sentiment_analysis if s['sentiment'] == 'bullish']),
            "bearish_count": len([s for s in sentiment_analysis if s['sentiment'] == 'bearish']),
            "neutral_count": len([s for s in sentiment_analysis if s['sentiment'] == 'neutral'])
        }
    
    def _analyze_stock_sentiments(self, price_data: Dict, news_data: Dict) -> List[Dict]:
        """Analyze sentiment for individual stocks"""
        sentiments = []
        gainers = price_data.get('top_gainers', [])
        losers = price_data.get('top_losers', [])
        all_movers = gainers + losers
        
        if not all_movers:
            return []
        
        sentiment_prompt = f"""
        Analyze sentiment for these stocks based on price movements:
        
        {safe_json_dumps(all_movers[:10], indent=2)}
        
        For each stock, determine sentiment as 'bullish', 'bearish', or 'neutral' based on:
        - Daily return percentage
        - Volume ratio
        - Price momentum
        
        Return JSON array:
        {{
            "sentiments": [
                {{"symbol": "RELIANCE", "sentiment": "bullish", "confidence": 0.8, "reason": "strong volume breakout"}}
            ]
        }}
        """
        
        response = self._llm_call(sentiment_prompt)
        
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1:
                result = json.loads(response[json_start:json_end])
                return result.get('sentiments', [])
        except:
            pass
        
        # Fallback: basic sentiment from returns
        for mover in all_movers[:20]:
            if mover['daily_return'] > 2:
                sentiment = 'bullish'
            elif mover['daily_return'] < -2:
                sentiment = 'bearish'
            else:
                sentiment = 'neutral'
            
            sentiments.append({
                'symbol': mover['symbol'],
                'sentiment': sentiment,
                'confidence': min(abs(mover['daily_return']) / 10, 1.0),
                'reason': f"{mover['daily_return']:.1f}% movement"
            })
        
        return sentiments
    
    def _calculate_market_sentiment(self, stock_sentiments: List[Dict]) -> Dict:
        """Calculate overall market sentiment"""
        if not stock_sentiments:
            return {"overall": "neutral", "confidence": 0.5}
        
        bullish = len([s for s in stock_sentiments if s['sentiment'] == 'bullish'])
        bearish = len([s for s in stock_sentiments if s['sentiment'] == 'bearish'])
        neutral = len([s for s in stock_sentiments if s['sentiment'] == 'neutral'])
        total = len(stock_sentiments)
        
        if bullish > bearish * 1.5:
            overall = "bullish"
        elif bearish > bullish * 1.5:
            overall = "bearish"
        else:
            overall = "neutral"
        
        return {
            "overall": overall,
            "confidence": max(bullish, bearish) / total,
            "bullish_pct": bullish / total * 100,
            "bearish_pct": bearish / total * 100,
            "neutral_pct": neutral / total * 100
        }

class PriceAnalysisAgent(BaseAgent):
    def __init__(self, llm_client):
        super().__init__("Price_Agent", llm_client)
    
    def execute(self, input_data: Dict) -> Dict:
        """Analyze price movements for stocks from CSV"""
        stock_symbols = input_data.get('stock_symbols', [])
        stock_data = self._fetch_price_data(stock_symbols)
        movers = self._identify_significant_movers(stock_data)
        
        return {
            "agent": self.name,
            "top_gainers": movers['gainers'][:10],
            "top_losers": movers['losers'][:10],
            "total_analyzed": len(stock_data),
            "market_depth": self._calculate_market_depth(movers)
        }
    
    def _fetch_price_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch stock price data"""
        stock_data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                if not hist.empty and len(hist) >= 2:
                    hist['Daily_Return'] = hist['Close'].pct_change()
                    hist['Volume_MA'] = hist['Volume'].rolling(window=3).mean()
                    hist['Price_Change'] = hist['Close'] - hist['Close'].shift(1)
                    stock_data[symbol] = hist
            except Exception as e:
                continue
        
        return stock_data
    
    def _identify_significant_movers(self, stock_data: Dict) -> Dict:
        """Identify significant price movements"""
        movers = {'gainers': [], 'losers': []}
        
        for symbol, data in stock_data.items():
            if len(data) < 2:
                continue
            
            latest_return = data['Daily_Return'].iloc[-1]
            latest_close = data['Close'].iloc[-1]
            volume_ratio = data['Volume'].iloc[-1] / data['Volume_MA'].iloc[-1] if data['Volume_MA'].iloc[-1] > 0 else 1
            
            stock_info = {
                'symbol': symbol.replace('.NS', ''),
                'daily_return': float(latest_return * 100),
                'price_change': float(data['Price_Change'].iloc[-1]),
                'current_price': float(latest_close),
                'volume_ratio': float(volume_ratio),
                'volume': int(data['Volume'].iloc[-1])
            }
            
            if latest_return > 0.01:  # >1% gain
                movers['gainers'].append(stock_info)
            elif latest_return < -0.01:  # >1% loss
                movers['losers'].append(stock_info)
        
        movers['gainers'].sort(key=lambda x: x['daily_return'], reverse=True)
        movers['losers'].sort(key=lambda x: x['daily_return'])
        
        return movers
    
    def _calculate_market_depth(self, movers: Dict) -> Dict:
        """Calculate market depth metrics"""
        gainers = movers['gainers']
        losers = movers['losers']
        
        return {
            "advancing": len(gainers),
            "declining": len(losers),
            "advance_decline_ratio": len(gainers) / len(losers) if losers else len(gainers),
            "strong_gainers": len([g for g in gainers if g['daily_return'] > 3]),
            "strong_losers": len([l for l in losers if l['daily_return'] < -3])
        }

class RSSAgent(BaseAgent):
    def __init__(self, llm_client):
        super().__init__("RSS_Agent", llm_client)
        self.feeds = {
            "Market News": [
                "https://www.livemint.com/rss/markets",
                "https://www.zeebiz.com/india-markets.xml"
            ]
        }
    
    def execute(self, input_data: Dict) -> Dict:
        """Fetch and filter market news"""
        raw_news = self._fetch_rss_feeds()
        filtered_news = self._filter_relevant_news(raw_news)
        
        return {
            "agent": self.name,
            "filtered_news": filtered_news[:10],
            "news_count": len(filtered_news)
        }
    
    def _fetch_rss_feeds(self) -> Dict[str, List[Dict]]:
        """Fetch RSS data"""
        all_news = {}
        today = datetime.now().date()
        
        for category, feeds in self.feeds.items():
            category_news = []
            for feed_url in feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:10]:
                        try:
                            published_date = datetime(*entry.published_parsed[:6]).date()
                            if published_date == today:
                                category_news.append({
                                    'title': entry.title,
                                    'summary': entry.get('summary', ''),
                                    'category': category
                                })
                        except:
                            continue
                except:
                    continue
            
            all_news[category] = category_news
        
        return all_news
    
    def _filter_relevant_news(self, raw_news: Dict) -> List[Dict]:
        """Filter market relevant news"""
        all_articles = []
        for category, articles in raw_news.items():
            all_articles.extend(articles)
        
        return all_articles[:15]  # Simple filtering for demo

class StaticHTMLGenerator:
    def __init__(self):
        self.llm_client = ollama.Client()
        self.csv_agent = CSVDataAgent(self.llm_client)
        self.price_agent = PriceAnalysisAgent(self.llm_client)
        self.rss_agent = RSSAgent(self.llm_client)
        self.sentiment_agent = SentimentAgent(self.llm_client)
    
    def execute_analysis(self) -> Dict:
        """Execute complete analysis pipeline"""
        print("Starting market analysis...")
        
        # 1. Load CSV data
        print("CSV Agent: Loading stock symbols...")
        csv_result = self.csv_agent.execute({})
        
        # 2. Price analysis
        print("Price Agent: Analyzing movements...")
        price_result = self.price_agent.execute(csv_result)
        
        # 3. News analysis
        print("RSS Agent: Collecting news...")
        news_result = self.rss_agent.execute({})
        
        # 4. Sentiment analysis
        print("Sentiment Agent: Analyzing sentiment...")
        sentiment_input = {
            'price_data': price_result,
            'news_data': news_result
        }
        sentiment_result = self.sentiment_agent.execute(sentiment_input)
        
        # 5. Generate market summary
        market_summary = self._generate_market_summary(price_result, sentiment_result)
        
        analysis_result = {
            'csv_data': csv_result,
            'price_data': price_result,
            'news_data': news_result,
            'sentiment_data': sentiment_result,
            'market_summary': market_summary,
            'timestamp': datetime.now().isoformat()
        }
        
        print("Analysis complete.")
        return analysis_result
    
    def _generate_market_summary(self, price_data: Dict, sentiment_data: Dict) -> str:
        """Generate market summary"""
        gainers_count = len(price_data.get('top_gainers', []))
        losers_count = len(price_data.get('top_losers', []))
        market_sentiment = sentiment_data.get('market_sentiment', {}).get('overall', 'neutral')
        
        summary = f"""
        <h4>Market Overview - {datetime.now().strftime('%Y-%m-%d')}</h4>
        <p><strong>Market Sentiment:</strong> {market_sentiment.upper()}</p>
        <p><strong>Active Stocks:</strong> {gainers_count} gainers, {losers_count} losers</p>
        <p><strong>Sentiment Distribution:</strong> 
           {sentiment_data.get('bullish_count', 0)} bullish, 
           {sentiment_data.get('bearish_count', 0)} bearish, 
           {sentiment_data.get('neutral_count', 0)} neutral</p>
        """
        
        if price_data.get('top_gainers'):
            top_gainer = price_data['top_gainers'][0]
            summary += f"<p><strong>Top Gainer:</strong> {top_gainer['symbol']} (+{top_gainer['daily_return']:.2f}%)</p>"
        
        if price_data.get('top_losers'):
            top_loser = price_data['top_losers'][0]
            summary += f"<p><strong>Top Loser:</strong> {top_loser['symbol']} ({top_loser['daily_return']:.2f}%)</p>"
        
        return summary
    
    def generate_html_report(self, analysis_data: Dict, filename: str = "market_dashboard.html") -> str:
        """Generate static HTML file with analysis results"""
        
        html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Indian Stock Market Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .timestamp {{ text-align: center; color: #666; margin-bottom: 20px; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
        .card {{ border: 1px solid #ddd; padding: 20px; border-radius: 8px; background: #fafafa; }}
        .metric {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
        .bullish {{ color: #22c55e; }}
        .bearish {{ color: #ef4444; }}
        .neutral {{ color: #6b7280; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f8f9fa; font-weight: bold; }}
        .positive {{ color: #22c55e; font-weight: bold; }}
        .negative {{ color: #ef4444; font-weight: bold; }}
        .news-item {{ margin: 10px 0; padding: 10px; background: white; border-left: 3px solid #059669; }}
        .chart-container {{ position: relative; height: 300px; }}
        .refresh-note {{ text-align: center; margin: 20px 0; padding: 10px; background: #e3f2fd; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📈 Indian Stock Market Dashboard</h1>
            <div class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
        
        <div class="refresh-note">
            <strong>Note:</strong> This is a static report. Run the Python script again to refresh data.
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>📊 Market Sentiment</h3>
                <div id="sentiment-overview"></div>
                <div class="chart-container">
                    <canvas id="sentimentChart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h3>📈 Market Depth</h3>
                <div id="market-depth"></div>
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>🚀 Top Gainers</h3>
                <div id="top-gainers"></div>
            </div>
            
            <div class="card">
                <h3>📉 Top Losers</h3>
                <div id="top-losers"></div>
            </div>
        </div>
        
        <div class="card">
            <h3>📰 Today's Market Analysis</h3>
            <div id="market-summary"></div>
        </div>
        
        <div class="card">
            <h3>📢 Recent News</h3>
            <div id="news-section"></div>
        </div>
    </div>

    <script>
        // Embedded data from analysis
        const marketData = {safe_json_dumps(analysis_data, indent=8)};
        
        function initializeDashboard() {{
            updateSentimentOverview();
            updateSentimentChart();
            updateMarketDepth();
            updateMoversTable('top-gainers', marketData.price_data?.top_gainers || []);
            updateMoversTable('top-losers', marketData.price_data?.top_losers || []);
            updateMarketSummary();
            updateNewsSection();
        }}
        
        function updateSentimentOverview() {{
            const sentiment = marketData.sentiment_data?.market_sentiment || {{}};
            document.getElementById('sentiment-overview').innerHTML = `
                <div class="metric ${{sentiment.overall || 'neutral'}}">${{(sentiment.overall || 'NEUTRAL').toUpperCase()}}</div>
                <p>Confidence: ${{((sentiment.confidence || 0) * 100).toFixed(1)}}%</p>
                <p>Bullish: ${{sentiment.bullish_pct?.toFixed(1) || 0}}% | 
                   Bearish: ${{sentiment.bearish_pct?.toFixed(1) || 0}}% | 
                   Neutral: ${{sentiment.neutral_pct?.toFixed(1) || 0}}%</p>
            `;
        }}
        
        function updateSentimentChart() {{
            const ctx = document.getElementById('sentimentChart').getContext('2d');
            const sentimentData = marketData.sentiment_data || {{}};
            
            const bullishCount = sentimentData.bullish_count || 0;
            const bearishCount = sentimentData.bearish_count || 0;
            const neutralCount = sentimentData.neutral_count || 0;
            
            if (bullishCount > 0 || bearishCount > 0 || neutralCount > 0) {{
                new Chart(ctx, {{
                    type: 'doughnut',
                    data: {{
                        labels: ['Bullish', 'Bearish', 'Neutral'],
                        datasets: [{{
                            data: [bullishCount, bearishCount, neutralCount],
                            backgroundColor: ['#22c55e', '#ef4444', '#6b7280']
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{
                                position: 'bottom'
                            }}
                        }}
                    }}
                }});
            }}
        }}
        
        function updateMarketDepth() {{
            const depth = marketData.price_data?.market_depth || {{}};
            document.getElementById('market-depth').innerHTML = `
                <p><strong>Advancing:</strong> ${{depth.advancing || 0}}</p>
                <p><strong>Declining:</strong> ${{depth.declining || 0}}</p>
                <p><strong>A/D Ratio:</strong> ${{(depth.advance_decline_ratio || 0).toFixed(2)}}</p>
                <p><strong>Strong Gainers (>3%):</strong> ${{depth.strong_gainers || 0}}</p>
                <p><strong>Strong Losers (<-3%):</strong> ${{depth.strong_losers || 0}}</p>
                <p><strong>Total Analyzed:</strong> ${{marketData.price_data?.total_analyzed || 0}} stocks</p>
            `;
        }}
        
        function updateMoversTable(elementId, movers) {{
            if (!Array.isArray(movers) || movers.length === 0) {{
                document.getElementById(elementId).innerHTML = '<p>No significant movers found</p>';
                return;
            }}
            
            const html = `
                <table>
                    <thead>
                        <tr><th>Symbol</th><th>Change %</th><th>Price (Rs)</th><th>Volume Ratio</th></tr>
                    </thead>
                    <tbody>
                        ${{movers.slice(0, 10).map(stock => `
                            <tr>
                                <td>${{stock.symbol || 'N/A'}}</td>
                                <td class="${{(stock.daily_return || 0) > 0 ? 'positive' : 'negative'}}">
                                    ${{(stock.daily_return || 0).toFixed(2)}}%
                                </td>
                                <td>Rs ${{(stock.current_price || 0).toFixed(2)}}</td>
                                <td>${{(stock.volume_ratio || 0).toFixed(2)}}x</td>
                            </tr>
                        `).join('')}}
                    </tbody>
                </table>
            `;
            document.getElementById(elementId).innerHTML = html;
        }}
        
        function updateMarketSummary() {{
            document.getElementById('market-summary').innerHTML = 
                marketData.market_summary || 'No analysis available';
        }}
        
        function updateNewsSection() {{
            const news = marketData.news_data?.filtered_news || [];
            if (news.length === 0) {{
                document.getElementById('news-section').innerHTML = '<p>No recent news available</p>';
                return;
            }}
            
            const html = news.map(article => `
                <div class="news-item">
                    <strong>${{article.title || 'No title'}}</strong>
                    <p>${{article.summary || 'No summary available'}}</p>
                    <small>Category: ${{article.category || 'General'}}</small>
                </div>
            `).join('');
            
            document.getElementById('news-section').innerHTML = html;
        }}
        
        // Initialize dashboard when page loads
        document.addEventListener('DOMContentLoaded', initializeDashboard);
    </script>
</body>
</html>
        '''
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML report generated: {filename}")
        return filename
    
    def run_analysis_and_generate_html(self):
        """Run complete analysis and generate HTML file"""
        analysis_data = self.execute_analysis()
        html_file = self.generate_html_report(analysis_data)
        
        print(f"\n{'='*50}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*50}")
        print(f"HTML Dashboard: {html_file}")
        print(f"Open the file in your browser to view the dashboard")
        
        return html_file

# Usage
if __name__ == "__main__":
    generator = StaticHTMLGenerator()
    generator.run_analysis_and_generate_html()