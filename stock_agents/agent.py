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

class RSSAgent(BaseAgent):
    def __init__(self, llm_client):
        super().__init__("RSS_Agent", llm_client)
        self.feeds = {
            "Market News": [
                "https://www.livemint.com/rss/markets",
                "https://www.zeebiz.com/india-markets.xml",
                "https://www.thehindubusinessline.com/markets/stock-markets/feeder/default.rss"
            ],
            "Stocks & Companies": [
                "https://www.livemint.com/rss/companies",
                "https://www.moneyworks4me.com/company/news/latest-stock-news-rss/company-news"
            ],
            "Losers and Gainers": [
                "https://www.thehindubusinessline.com/markets/top-gainers-and-top-losers/feeder/default.rss"
            ]
        }
    
    def execute(self, input_data: Dict) -> Dict:
        """Autonomously collect and filter relevant market news"""
        raw_news = self._fetch_rss_feeds()
        filtered_news = self._filter_relevant_news(raw_news)
        prioritized_news = self._prioritize_news(filtered_news)
        
        return {
            "agent": self.name,
            "filtered_news": prioritized_news,
            "news_count": len(prioritized_news),
            "categories": list(self.feeds.keys())
        }
    
    def _fetch_rss_feeds(self) -> Dict[str, List[Dict]]:
        """Fetch raw RSS data"""
        all_news = {}
        today = datetime.now().date()
        
        for category, feeds in self.feeds.items():
            category_news = []
            for feed_url in feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:15]:
                        try:
                            published_date = datetime(*entry.published_parsed[:6]).date()
                            if published_date == today:
                                category_news.append({
                                    'title': entry.title,
                                    'summary': entry.get('summary', ''),
                                    'link': entry.link,
                                    'published': entry.published,
                                    'category': category
                                })
                        except:
                            continue
                except Exception as e:
                    print(f"RSS fetch error {feed_url}: {e}")
            
            all_news[category] = category_news
        
        return all_news
    
    def _filter_relevant_news(self, raw_news: Dict) -> List[Dict]:
        """Use LLM to filter market-relevant news"""
        all_articles = []
        for category, articles in raw_news.items():
            all_articles.extend(articles)
        
        if not all_articles:
            return []
        
        # Create summary for LLM filtering
        news_summary = "\n".join([f"- {article['title']}" for article in all_articles[:20]])
        
        filter_prompt = f"""
        Filter these news articles for Indian stock market relevance. Return only articles that could impact stock prices.
        
        News Articles:
        {news_summary}
        
        Return JSON array with indices of relevant articles (0-based):
        {{"relevant_indices": [0, 2, 5]}}
        """
        
        response = self._llm_call(filter_prompt)
        
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1:
                result = json.loads(response[json_start:json_end])
                relevant_indices = result.get('relevant_indices', [])
                return [all_articles[i] for i in relevant_indices if i < len(all_articles)]
        except:
            pass
        
        # Fallback: return first 10 articles
        return all_articles[:10]
    
    def _prioritize_news(self, filtered_news: List[Dict]) -> List[Dict]:
        """Prioritize news by market impact potential"""
        if not filtered_news:
            return []
        
        priority_prompt = f"""
        Rank these news articles by potential stock market impact (1=highest, 5=lowest).
        
        Articles:
        {safe_json_dumps([{'title': news['title'], 'category': news['category']} for news in filtered_news], indent=2)}
        
        Return JSON:
        {{"rankings": [{{"index": 0, "priority": 1, "reason": "major market catalyst"}}]}}
        """
        
        response = self._llm_call(priority_prompt)
        
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1:
                result = json.loads(response[json_start:json_end])
                rankings = result.get('rankings', [])
                
                # Apply priorities
                for ranking in rankings:
                    idx = ranking.get('index')
                    if idx < len(filtered_news):
                        filtered_news[idx]['priority'] = ranking.get('priority', 5)
                        filtered_news[idx]['priority_reason'] = ranking.get('reason', '')
        except:
            # Fallback: assign default priorities
            for i, news in enumerate(filtered_news):
                news['priority'] = 3
                news['priority_reason'] = 'Default priority'
        
        return sorted(filtered_news, key=lambda x: x.get('priority', 5))

class PriceAnalysisAgent(BaseAgent):
    def __init__(self, llm_client):
        super().__init__("Price_Agent", llm_client)
        self.nse_tickers = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "HINDUNILVR.NS",
            "INFY.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "ASIANPAINT.NS",
            "HCLTECH.NS", "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS", "WIPRO.NS",
            "MARUTI.NS", "TITAN.NS", "NESTLEIND.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS"
        ]
    
    def execute(self, input_data: Dict) -> Dict:
        """Autonomously analyze price movements and identify significant movers"""
        stock_data = self._fetch_price_data()
        movers = self._identify_significant_movers(stock_data)
        movement_analysis = self._analyze_movements(movers, stock_data)
        
        return {
            "agent": self.name,
            "top_gainers": movement_analysis['gainers'],
            "top_losers": movement_analysis['losers'],
            "movement_patterns": movement_analysis['patterns'],
            "volume_signals": movement_analysis['volume_signals']
        }
    
    def _fetch_price_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch stock price data"""
        stock_data = {}
        for symbol in self.nse_tickers:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                if not hist.empty and len(hist) >= 2:
                    hist['Daily_Return'] = hist['Close'].pct_change()
                    hist['Volume_MA'] = hist['Volume'].rolling(window=3).mean()
                    hist['Price_Change'] = hist['Close'] - hist['Close'].shift(1)
                    stock_data[symbol] = hist
            except Exception as e:
                print(f"Price fetch error {symbol}: {e}")
        
        return stock_data
    
    def _identify_significant_movers(self, stock_data: Dict) -> Dict:
        """Identify stocks with significant price movements"""
        movers = {'gainers': [], 'losers': []}
        
        for symbol, data in stock_data.items():
            if len(data) < 2:
                continue
            
            latest_return = data['Daily_Return'].iloc[-1]
            latest_close = data['Close'].iloc[-1]
            volume_ratio = data['Volume'].iloc[-1] / data['Volume_MA'].iloc[-1] if data['Volume_MA'].iloc[-1] > 0 else 1
            
            stock_info = {
                'symbol': symbol.replace('.NS', ''),
                'full_symbol': symbol,
                'daily_return': float(latest_return * 100),
                'price_change': float(data['Price_Change'].iloc[-1]),
                'current_price': float(latest_close),
                'volume_ratio': float(volume_ratio),
                'volume': int(data['Volume'].iloc[-1])
            }
            
            # Threshold for significant movement
            if abs(latest_return) > 0.015:  # >1.5% movement
                if latest_return > 0:
                    movers['gainers'].append(stock_info)
                else:
                    movers['losers'].append(stock_info)
        
        # Sort by absolute return
        movers['gainers'].sort(key=lambda x: x['daily_return'], reverse=True)
        movers['losers'].sort(key=lambda x: x['daily_return'])
        
        return movers
    
    def _analyze_movements(self, movers: Dict, stock_data: Dict) -> Dict:
        """LLM analysis of price movement patterns"""
        top_movers = movers['gainers'][:5] + movers['losers'][:5]
        
        analysis_prompt = f"""
        Analyze these stock price movements for patterns and signals:
        
        {safe_json_dumps(top_movers, indent=2)}
        
        Identify:
        1. Movement patterns (momentum, reversal, breakout)
        2. Volume signals (unusual volume, volume confirmation)
        3. Sector clustering
        
        Return JSON:
        {{
            "patterns": {{"pattern_type": "description"}},
            "volume_signals": ["signal1", "signal2"],
            "sector_analysis": {{"sector": "movement_type"}}
        }}
        """
        
        response = self._llm_call(analysis_prompt)
        
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1:
                analysis = json.loads(response[json_start:json_end])
            else:
                analysis = {"patterns": {}, "volume_signals": [], "sector_analysis": {}}
        except:
            analysis = {"patterns": {}, "volume_signals": [], "sector_analysis": {}}
        
        return {
            'gainers': movers['gainers'][:5],
            'losers': movers['losers'][:5],
            'patterns': analysis.get('patterns', {}),
            'volume_signals': analysis.get('volume_signals', []),
            'sector_analysis': analysis.get('sector_analysis', {})
        }

class CorrelationAgent(BaseAgent):
    def __init__(self, llm_client):
        super().__init__("Correlation_Agent", llm_client)
    
    def execute(self, input_data: Dict) -> Dict:
        """Correlate news events with price movements"""
        news_data = input_data.get('news_data', {})
        price_data = input_data.get('price_data', {})
        
        correlations = self._find_news_price_correlations(news_data, price_data)
        catalysts = self._identify_catalysts(correlations)
        momentum_signals = self._analyze_momentum_signals(correlations, price_data)
        
        return {
            "agent": self.name,
            "correlations": correlations,
            "identified_catalysts": catalysts,
            "momentum_signals": momentum_signals,
            "sector_trends": self._analyze_sector_trends(correlations)
        }
    
    def _find_news_price_correlations(self, news_data: Dict, price_data: Dict) -> List[Dict]:
        """Match news events with stock price movements"""
        correlations = []
        
        filtered_news = news_data.get('filtered_news', [])
        gainers = price_data.get('top_gainers', [])
        losers = price_data.get('top_losers', [])
        
        all_movers = gainers + losers
        
        correlation_prompt = f"""
        Match news events with stock price movements:
        
        News Events:
        {safe_json_dumps([{'title': news['title'], 'category': news['category']} for news in filtered_news[:10]], indent=2)}
        
        Stock Movements:
        {safe_json_dumps([{'symbol': mover['symbol'], 'return': mover['daily_return']} for mover in all_movers[:10]], indent=2)}
        
        Return correlations as JSON:
        {{
            "correlations": [
                {{"news_index": 0, "stock_symbol": "RELIANCE", "correlation_strength": "high", "reason": "earnings beat"}}
            ]
        }}
        """
        
        response = self._llm_call(correlation_prompt)
        
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1:
                result = json.loads(response[json_start:json_end])
                raw_correlations = result.get('correlations', [])
                
                for corr in raw_correlations:
                    news_idx = corr.get('news_index')
                    if news_idx < len(filtered_news):
                        correlations.append({
                            'news': filtered_news[news_idx],
                            'stock_symbol': corr.get('stock_symbol'),
                            'correlation_strength': corr.get('correlation_strength'),
                            'reason': corr.get('reason'),
                            'movement_data': next((m for m in all_movers if m['symbol'] == corr.get('stock_symbol')), None)
                        })
        except Exception as e:
            print(f"Correlation analysis error: {e}")
        
        return correlations
    
    def _identify_catalysts(self, correlations: List[Dict]) -> List[Dict]:
        """Identify key market catalysts"""
        if not correlations:
            return []
        
        catalyst_prompt = f"""
        Identify key market catalysts from these correlations:
        
        {safe_json_dumps([{'reason': c['reason'], 'strength': c['correlation_strength']} for c in correlations], indent=2)}
        
        Return top catalysts:
        {{
            "catalysts": [
                {{"catalyst": "earnings season", "impact": "high", "affected_sectors": ["IT", "Banking"]}}
            ]
        }}
        """
        
        response = self._llm_call(catalyst_prompt)
        
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1:
                result = json.loads(response[json_start:json_end])
                return result.get('catalysts', [])
        except:
            pass
        
        return []
    
    def _analyze_momentum_signals(self, correlations: List[Dict], price_data: Dict) -> Dict:
        """Analyze momentum continuation signals"""
        momentum_prompt = f"""
        Based on these price movements and correlations, predict 2-3 day momentum:
        
        Price Data: {json.dumps(price_data.get('movement_patterns', {}), indent=2)}
        Volume Signals: {json.dumps(price_data.get('volume_signals', []), indent=2)}
        
        Return momentum forecast:
        {{
            "momentum_forecast": {{"RELIANCE": "bullish_continuation", "TCS": "consolidation"}},
            "key_levels": {{"RELIANCE": "resistance_at_2500"}},
            "risk_factors": ["global_uncertainty", "sector_rotation"]
        }}
        """
        
        response = self._llm_call(momentum_prompt)
        
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1:
                return json.loads(response[json_start:json_end])
        except:
            pass
        
        return {"momentum_forecast": {}, "key_levels": {}, "risk_factors": []}
    
    def _analyze_sector_trends(self, correlations: List[Dict]) -> Dict:
        """Identify sector-wide trends"""
        sector_mapping = {
            'RELIANCE': 'Energy', 'TCS': 'IT', 'INFY': 'IT', 'HDFCBANK': 'Banking',
            'ICICIBANK': 'Banking', 'SBIN': 'Banking', 'MARUTI': 'Auto', 'TITAN': 'Consumer'
        }
        
        sector_movements = {}
        for corr in correlations:
            symbol = corr.get('stock_symbol', '')
            sector = sector_mapping.get(symbol, 'Other')
            movement_data = corr.get('movement_data')
            
            if movement_data and sector not in sector_movements:
                sector_movements[sector] = []
            
            if movement_data:
                sector_movements[sector].append(movement_data['daily_return'])
        
        sector_trends = {}
        for sector, returns in sector_movements.items():
            if returns:
                avg_return = sum(returns) / len(returns)
                sector_trends[sector] = "bullish" if avg_return > 1 else "bearish" if avg_return < -1 else "neutral"
        
        return sector_trends

class ReportAgent(BaseAgent):
    def __init__(self, llm_client):
        super().__init__("Report_Agent", llm_client)
    
    def execute(self, input_data: Dict) -> Dict:
        """Generate comprehensive market report and Instagram content"""
        report = self._generate_market_report(input_data)
        chart_path = self._create_instagram_chart(input_data)
        insights = self._generate_key_insights(input_data)
        
        return {
            "agent": self.name,
            "market_report": report,
            "instagram_chart": chart_path,
            "key_insights": insights,
            "executive_summary": self._create_executive_summary(input_data)
        }
    
    def _generate_market_report(self, data: Dict) -> str:
        """Generate comprehensive market report"""
        correlation_data = data.get('correlation_data', {})
        price_data = data.get('price_data', {})
        
        report = f"""
# Daily Indian Stock Market Analysis Report
**Date:** {datetime.now().strftime('%Y-%m-%d')}

## Executive Summary
{self._create_executive_summary(data)}

## Top Movers

### Gainers
"""
        for gainer in price_data.get('top_gainers', [])[:5]:
            report += f"- **{gainer['symbol']}**: +{gainer['daily_return']:.2f}% (₹{gainer['current_price']:.2f})\n"
        
        report += "\n### Losers\n"
        for loser in price_data.get('top_losers', [])[:5]:
            report += f"- **{loser['symbol']}**: {loser['daily_return']:.2f}% (₹{loser['current_price']:.2f})\n"
        
        report += f"""
## Key Market Catalysts
{chr(10).join(['- ' + catalyst.get('catalyst', '') for catalyst in correlation_data.get('identified_catalysts', [])])}

## Sector Trends
{chr(10).join([f"- **{sector}**: {trend}" for sector, trend in correlation_data.get('sector_trends', {}).items()])}

## Momentum Forecast (2-3 Days)
{chr(10).join([f"- **{stock}**: {forecast}" for stock, forecast in correlation_data.get('momentum_signals', {}).get('momentum_forecast', {}).items()])}

## Key Insights
{chr(10).join(['- ' + insight for insight in self._generate_key_insights(data)])}
"""
        
        return report
    
    def _create_executive_summary(self, data: Dict) -> str:
        """Create executive summary using LLM"""
        summary_prompt = f"""
        Create a 2-sentence executive summary of today's Indian stock market:
        
        Top Gainers: {data.get('price_data', {}).get('top_gainers', [])[:3]}
        Top Losers: {data.get('price_data', {}).get('top_losers', [])[:3]}
        Key Catalysts: {data.get('correlation_data', {}).get('identified_catalysts', [])}
        
        Format: "Market overview sentence. Key driver sentence."
        """
        
        response = self._llm_call(summary_prompt)
        return response.strip()
    
    def _generate_key_insights(self, data: Dict) -> List[str]:
        """Generate actionable market insights"""
        insights_prompt = f"""
        Generate 3 key actionable insights for traders:
        
        Data: {json.dumps(data, default=str, indent=2)[:1000]}
        
        Return as JSON array:
        {{"insights": ["insight1", "insight2", "insight3"]}}
        """
        
        response = self._llm_call(insights_prompt)
        
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1:
                result = json.loads(response[json_start:json_end])
                return result.get('insights', [])
        except:
            pass
        
        return ["Market analysis completed", "Monitor key sectors", "Watch volume signals"]
    
    def _create_instagram_chart(self, data: Dict) -> str:
        """Create Instagram-ready market chart"""
        price_data = data.get('price_data', {})
        gainers = price_data.get('top_gainers', [])[:5]
        losers = price_data.get('top_losers', [])[:5]
        
        if not gainers and not losers:
            return "No chart data available"
        
        # Create output directory
        import os
        output_dir = "output/charts"
        os.makedirs(output_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Top gainers chart
        if gainers:
            gainer_symbols = [g['symbol'] for g in gainers]
            gainer_returns = [g['daily_return'] for g in gainers]
            
            bars1 = ax1.barh(gainer_symbols, gainer_returns, color='green', alpha=0.7)
            ax1.set_title('Top 5 Gainers', fontsize=16, fontweight='bold')
            ax1.set_xlabel('Daily Return (%)')
            
            for bar, value in zip(bars1, gainer_returns):
                ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                        f'{value:.1f}%', va='center')
        
        # Top losers chart
        if losers:
            loser_symbols = [l['symbol'] for l in losers]
            loser_returns = [l['daily_return'] for l in losers]
            
            bars2 = ax2.barh(loser_symbols, loser_returns, color='red', alpha=0.7)
            ax2.set_title('Top 5 Losers', fontsize=16, fontweight='bold')
            ax2.set_xlabel('Daily Return (%)')
            
            for bar, value in zip(bars2, loser_returns):
                ax2.text(bar.get_width() - 0.1, bar.get_y() + bar.get_height()/2, 
                        f'{value:.1f}%', va='center', ha='right')
        
        plt.tight_layout()
        chart_path = os.path.join(output_dir, f"market_chart_{datetime.now().strftime('%Y%m%d_%H%M')}.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path

class MarketResearchOrchestrator:
    def __init__(self):
        self.llm_client = ollama.Client()
        self.rss_agent = RSSAgent(self.llm_client)
        self.price_agent = PriceAnalysisAgent(self.llm_client)
        self.correlation_agent = CorrelationAgent(self.llm_client)
        self.report_agent = ReportAgent(self.llm_client)
    
    def execute_daily_analysis(self) -> Dict:
        """Execute multi-agent analysis pipeline"""
        print("Starting multi-agent market analysis...")
        
        # Agent 1: RSS news collection and filtering
        print("RSS Agent: Collecting market news...")
        news_result = self.rss_agent.execute({})
        
        # Agent 2: Price movement analysis
        print("Price Agent: Analyzing stock movements...")
        price_result = self.price_agent.execute({})
        
        # Agent 3: News-price correlation analysis
        print("Correlation Agent: Finding catalysts...")
        correlation_input = {
            'news_data': news_result,
            'price_data': price_result
        }
        correlation_result = self.correlation_agent.execute(correlation_input)
        
        # Agent 4: Report generation
        print("Report Agent: Generating final report...")
        report_input = {
            'news_data': news_result,
            'price_data': price_result,
            'correlation_data': correlation_result
        }
        report_result = self.report_agent.execute(report_input)
        
        return {
            'news_analysis': news_result,
            'price_analysis': price_result,
            'correlation_analysis': correlation_result,
            'final_report': report_result,
            'pipeline_status': 'completed'
        }

# Usage
if __name__ == "__main__":
    orchestrator = MarketResearchOrchestrator()
    results = orchestrator.execute_daily_analysis()
    
    print("\n" + "="*50)
    print("FINAL MARKET REPORT")
    print("="*50)
    print(results['final_report']['market_report'])
    print(f"\nChart saved: {results['final_report']['instagram_chart']}")
    print(f"\nKey Insights: {results['final_report']['key_insights']}")