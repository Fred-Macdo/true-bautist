from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from typing import List, Dict, Callable, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class MarketCondition:
    """Market condition assessment results"""
    trend: str  # bullish, bearish, neutral
    volatility: str  # high, low, normal
    breadth: str  # expanding, contracting
    sentiment: str  # positive, negative, neutral
    risk_level: str  # high, medium, low
    details: Dict  # detailed metrics

class MarketScanner:

    def __init__(self, api_key: str,
             api_secret: str, 
             watchlist: List[str],
             timeframe: TimeFrame,
             start,  
             end):
    
        """Initialize scanner with Alpaca V2 API"""
        self.data_client = StockHistoricalDataClient(api_key, api_secret)
        self.watchlist: List[str] = []
        self.screening_criteria: Dict[str, Callable] = {}
        # Remove trailing commas to prevent tuple creation
        self.timeframe = timeframe
        self.starttime = start
        self.endtime = end

        # Market indices for macro analysis
        self.macro_symbols = {
            'indices': ['SPY', 'QQQ', 'IWM'],  # S&P 500, Nasdaq, Russell 2000
            'sectors': ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY'],  # Tech, Finance, Energy, Healthcare, Industrial, Consumer Staples, Consumer Discretionary
            'volatility': ['UVXY'],  # Volatility indices
            'rates': ['TLT', 'IEF'],  # 20+ year Treasury, 7-10 year Treasury
            'commodities': ['GLD', 'SLV', 'USO']  # Gold, Silver, Oil
        }
        
        # Russell-2000 stocks organized by sector
        self.iwm_stocks = [
            # Financial Services
            'STBA', 'CASH', 'HOPE', 'SBCF', 'BANR', 'FIBK', 'SFBS', 'HTLF', 'UBSI', 'PNFP',
            
            # Healthcare
            'OMCL', 'MMSI', 'MEDP', 'SRDX', 'LNTH', 'ICUI', 'NVRO', 'IRTC', 'TNDM', 'VCYT',
            
            # Technology
            'POWI', 'SMTC', 'FORM', 'CCMP', 'ONTO', 'DIOD', 'SYNA', 'MKSI', 'BRKS', 'NOVT',
            
            # Consumer Discretionary
            'BOOT', 'PLCE', 'CROX', 'DKS', 'FIVE', 'DECK', 'TXRH', 'PLAY', 'BJRI', 'CAKE',
            
            # Industrials
            'AIRC', 'MATX', 'AEIS', 'HLIO', 'GTLS', 'B', 'AGCO', 'KBR', 'HUBB', 'GGG',
            
            # Materials
            'CLF', 'MP', 'STLD', 'WOR', 'CRS', 'KRA', 'SCL', 'CBT', 'CC', 'ASIX',
            
            # Energy
            'PDCE', 'SM', 'MTDR', 'RRC', 'CNX', 'CTRA', 'CHX', 'HP', 'NEX', 'WTTR',
            
            # Real Estate
            'CSR', 'UE', 'BRX', 'RPT', 'KRG', 'SITC', 'ROIC', 'ADC', 'STAG', 'DEI',
            
            # Utilities
            'SPWR', 'BE', 'FSLR', 'NOVA', 'SEDG', 'ENPH', 'RUN', 'DQ', 'JKS', 'CSIQ',
            
            # Communication Services
            'YELP', 'APPS', 'BAND', 'CARS', 'ZNGA', 'TTGT', 'EVER', 'MTCH', 'GDDY', 'TTD'
        ]

        # Nasdaq-100 stocks organized by sector
        self.qqq_stocks = [
            # Technology
            'AAPL', 'MSFT', 'NVDA', 'AVGO', 'AMD', 'ADBE', 'CSCO', 'CRM', 'INTC', 'QCOM',
            'AMAT', 'ADI', 'PYPL', 'MU', 'INTU', 'KLAC', 'LRCX', 'SNPS', 'CDNS', 'NXPI',
            'MCHP', 'PANW', 'FTNT', 'ADSK', 'CTSH', 'TEAM', 'CPRT', 'ON', 'KEYS', 'SPLK',
            
            # Communication Services
            'META', 'GOOGL', 'GOOG', 'NFLX', 'CMCSA', 'TMUS', 'ATVI', 'EA', 'TTD', 'WBD',
            
            # Consumer Discretionary
            'AMZN', 'TSLA', 'PDD', 'BKNG', 'ABNB', 'ORLY', 'MAR', 'EBAY', 'DLTR', 'ROST',
            'NKE', 'SBUX', 'LULU', 'MELI', 'JD', 'LCID',
            
            # Healthcare
            'REGN', 'GILD', 'VRTX', 'ISRG', 'MRNA', 'DXCM', 'SGEN', 'ILMN', 'ALGN', 'IDXX',
            
            # Consumer Staples
            'PEP', 'COST', 'MDLZ', 'KHC', 'KDP', 'WBA',
            
            # Industrials
            'HON', 'CSX', 'PAYX', 'ODFL', 'FAST', 'CTAS', 'AEP', 'EXC',
            
            # Financial Services
            'NASDAQ', 'MSCI', 'VRSK', 'NDAQ',
            
            # Utilities & Energy
            'AEP', 'EXC', 'FANG',
            
            # Transportation
            'ODFL', 'LRCX', 'KLAC'
        ]

    def set_watchlist_to_iwm(self):
        """Set the watchlist to IWM (Russell 2000) stocks"""
        self.watchlist = self.iwm_stocks.copy()
        return self.watchlist

    def set_watchlist_to_qqq(self):
        """Set the watchlist to QQQ (Nasdaq-100) stocks"""
        self.watchlist = self.qqq_stocks.copy()
        return self.watchlist

    async def analyze_macro_environment(self) -> MarketCondition:
        """Analyze overall market conditions"""
        try:
            # Get data for all macro symbols
            all_symbols = [sym for syms in self.macro_symbols.values() for sym in syms]
            data = self._get_stock_data(
                all_symbols,
                start=self.starttime,
                timeframe=self.timeframe,
                end=self.endtime
            )
            
            for symbol in data.keys():
                # Drop duplicates while keeping the last occurrence
                data[symbol] = data[symbol].loc[~data[symbol].index.duplicated(keep='last')]
                
                # Ensure the index is sorted
                data[symbol] = data[symbol].sort_index()

            # Analyze market trend
            trend = self._analyze_market_trend(data)
            
            # Analyze market volatility
            volatility = self._analyze_volatility(data)
            
            # Analyze market breadth
            breadth = self._analyze_market_breadth(data)
            
            # Analyze market sentiment
            sentiment = self._analyze_sentiment(data)
            
            # Calculate risk level
            risk_level = self._calculate_risk_level(trend, volatility, breadth, sentiment)
            
            # Compile detailed metrics
            details = {
                'trend_metrics': trend['details'],
                'volatility_metrics': volatility['details'],
                'breadth_metrics': breadth['details'],
                'sentiment_metrics': sentiment['details'],
                'sector_performance': self._analyze_sector_performance(data),
                'correlation_matrix': self._calculate_correlation_matrix(data)
            }
            
            return MarketCondition(
                trend=trend['overall'],
                volatility=volatility['overall'],
                breadth=breadth['overall'],
                sentiment=sentiment['overall'],
                risk_level=risk_level,
                details=details
            )
            
        except Exception as e:
            print(f"Error analyzing macro environment: {e}")
            return None

    def _get_stock_data(self, symbols: List[str],  
                      start: datetime, 
                      timeframe: TimeFrame,
                      end: datetime = None) -> Dict[str, pd.DataFrame]:
        """
        Get historical stock data from Alpaca
        
        Args:
            symbols: List of stock symbols
            timeframe: Alpaca TimeFrame object
            start: Start datetime
            end: End datetime (defaults to now)
            
        Returns:
            Dictionary of DataFrames with OHLCV data for each symbol
        """
        try:
            if end is None:
                end = datetime.now()

            bars = self.data_client.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=symbols,
                    start=start,
                    end=end,
                    timeframe=timeframe
                    )
                )
            
            # Convert to dictionary of DataFrames
            data_dict = {}
            for symbol in symbols:
                if symbol in bars.data:
                    # Convert to DataFrame and unpack the data
                    df = pd.DataFrame([
                        {
                            'timestamp': bar.timestamp,
                            'open': bar.open,
                            'high': bar.high,
                            'low': bar.low,
                            'close': bar.close,
                            'volume': bar.volume,
                            'trade_count': bar.trade_count,
                            'vwap': bar.vwap
                        } for bar in bars.data[symbol]
                    ])
                    
                    # Convert timestamp to datetime and set as index
                    df.index = pd.to_datetime(df['timestamp'])
                    data_dict[symbol] = df

            return data_dict

        except Exception as e:
            print(f"Error getting stock data: {e}")
            return {}

    def _analyze_market_trend(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze overall market trend using multiple indicators"""
        try:
            spy_data = data['SPY']
            
            # Calculate moving averages
            spy_data['SMA50'] = spy_data['close'].rolling(window=50).mean()
            spy_data['SMA200'] = spy_data['close'].rolling(window=200).mean()
            
            # Calculate trend metrics
            price = spy_data['close'].iloc[-1]
            sma50 = spy_data['SMA50'].iloc[-1]
            sma200 = spy_data['SMA200'].iloc[-1]
            
            # Calculate momentum
            roc = (price / spy_data['close'].iloc[-20] - 1) * 100
            
            # Determine trend
            if price > sma50 and sma50 > sma200 and roc > 0:
                trend = 'bullish'
            elif price < sma50 and sma50 < sma200 and roc < 0:
                trend = 'bearish'
            else:
                trend = 'neutral'
                
            return {
                'overall': trend,
                'details': {
                    'price_vs_sma50': round((price / sma50 - 1) * 100, 2),  # Percentage
                    'price_vs_sma200': round((price / sma200 - 1) * 100, 2),  # Percentage
                    'momentum': roc,  # Already in percentage
                    'trend_strength': round(abs(roc), 2)
                }
            }
            
        except Exception as e:
            print(f"Error analyzing market trend: {e}")
            return {'overall': 'neutral', 'details': {}}

    def _analyze_volatility(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze market volatility"""
        try:
            vix_data = data['UVXY']
            
            # Calculate volatility metrics
            current_vix = vix_data['close'].iloc[-1]
            vix_avg = vix_data['close'].rolling(window=20).mean().iloc[-1]
            vix_percentile = self._calculate_percentile(vix_data['close'], current_vix)
            
            # Determine volatility level
            if current_vix > 30:
                volatility = 'high'
            elif current_vix < 15:
                volatility = 'low'
            else:
                volatility = 'normal'
                
            return {
                'overall': volatility,
                'details': {
                    'current_vix': current_vix,
                    'vix_20d_avg': vix_avg,
                    'vix_percentile': vix_percentile,
                    'vix_trend': current_vix / vix_avg - 1
                }
            }
            
        except Exception as e:
            print(f"Error analyzing volatility: {e}")
            return {'overall': 'normal', 'details': {}}

    def _analyze_market_breadth(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze market breadth using sector performance"""
        try:
            sector_returns = {}
            advancing_sectors = 0
            total_sectors = len(self.macro_symbols['sectors'])
            
            for sector in self.macro_symbols['sectors']:
                sector_data = data[sector]
                returns = round((sector_data['close'].iloc[-1] / sector_data['close'].iloc[0] - 1) * 100, 2)
                sector_returns[sector] = returns
                if returns > 0:
                    advancing_sectors += 1
            
            # Calculate breadth ratio as percentage
            breadth_ratio = round((advancing_sectors / total_sectors) * 100, 2)
            
            if breadth_ratio > 60:  # Changed from 0.6 to 60
                breadth = 'expanding'
            elif breadth_ratio < 40:  # Changed from 0.4 to 40
                breadth = 'contracting'
            else:
                breadth = 'neutral'
                
            return {
                'overall': breadth,
                'details': {
                    'advancing_sectors': advancing_sectors,
                    'total_sectors': total_sectors,
                    'breadth_ratio': breadth_ratio,  # Now in percentage
                    'sector_returns': sector_returns  # Already in percentage
                }
            }
            
        except Exception as e:
            print(f"Error analyzing market breadth: {e}")
            return {'overall': 'neutral', 'details': {}}

    def _analyze_sentiment(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze market sentiment using multiple indicators"""
        try:
            # Analyze trends (convert to percentages)
            vix_trend = round((data['UVXY']['close'].iloc[-1] / 
                      data['UVXY']['close'].iloc[-5] - 1) * 100, 2)
            
            gold_trend = round((data['GLD']['close'].iloc[-1] / 
                       data['GLD']['close'].iloc[-5] - 1) * 100, 2)
            
            treasury_trend = round((data['TLT']['close'].iloc[-1] / 
                          data['TLT']['close'].iloc[-5] - 1) * 100, 2)
            
            # Calculate sentiment score (now in percentage scale)
            sentiment_score = round(-vix_trend - (gold_trend + treasury_trend) / 2, 2)
            
            if sentiment_score > 2:  # Changed from 0.02 to 2
                sentiment = 'positive'
            elif sentiment_score < -2:  # Changed from -0.02 to -2
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
                
            return {
                'overall': sentiment,
                'details': {
                    'sentiment_score': sentiment_score,  # Now in percentage
                    'vix_trend': vix_trend,  # Now in percentage
                    'gold_trend': gold_trend,  # Now in percentage
                    'treasury_trend': treasury_trend  # Now in percentage
                }
            }
            
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {'overall': 'neutral', 'details': {}}

    def _calculate_risk_level(
        self,
        trend: Dict,
        volatility: Dict,
        breadth: Dict,
        sentiment: Dict
    ) -> str:
        """Calculate overall market risk level"""
        risk_score = 0
        
        # Add risk based on trend
        if trend['overall'] == 'bearish':
            risk_score += 2
        elif trend['overall'] == 'neutral':
            risk_score += 1
            
        # Add risk based on volatility
        if volatility['overall'] == 'high':
            risk_score += 2
        elif volatility['overall'] == 'normal':
            risk_score += 1
            
        # Add risk based on breadth
        if breadth['overall'] == 'contracting':
            risk_score += 2
        elif breadth['overall'] == 'neutral':
            risk_score += 1
            
        # Add risk based on sentiment
        if sentiment['overall'] == 'negative':
            risk_score += 2
        elif sentiment['overall'] == 'neutral':
            risk_score += 1
            
        # Determine risk level
        if risk_score >= 6:
            return 'high'
        elif risk_score >= 3:
            return 'medium'
        else:
            return 'low'

    def _analyze_sector_performance(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze relative performance of sectors"""
        sector_performance = {}
        spy_return = round((data['SPY']['close'].iloc[-1] / 
                    data['SPY']['close'].iloc[0] - 1) * 100, 2)
        
        for sector in self.macro_symbols['sectors']:
            sector_data = data[sector]
            sector_return = round((sector_data['close'].iloc[-1] / 
                          sector_data['close'].iloc[0] - 1) * 100, 2)
            sector_performance[sector] = {
                'absolute_return': sector_return,  # Now in percentage
                'relative_return': round(sector_return - spy_return, 2)  # Now in percentage
            }
            
        return sector_performance

    def scan_for_volume_breakouts(self, volume_threshold: float = 2.0, price_change_threshold: float = 2.0) -> Dict[str, Dict]:
        """
        Scan for stocks showing unusual volume activity indicating potential breakouts
        
        Args:
            volume_threshold: Multiplier for average volume (e.g., 2.0 = 200% of average volume)
            price_change_threshold: Minimum price change in percentage (e.g., 2.0 = 2%)
        """
        try:
            breakout_candidates = {}
            
            data = self._get_stock_data(
                self.watchlist,
                start=self.starttime,
                timeframe=self.timeframe,
                end=self.endtime
            )
            
            for symbol, df in data.items():
                # Calculate volume metrics
                avg_volume = df['volume'].rolling(window=20).mean()
                current_volume = df['volume'].iloc[-1]
                volume_ratio = round((current_volume / avg_volume.iloc[-1]) * 100, 2)
                
                # Calculate price metrics
                price_change = round((df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100, 2)
                
                # Check for breakout conditions
                if (volume_ratio > volume_threshold * 100 and 
                    abs(price_change) > price_change_threshold):
                    
                    # Calculate additional metrics
                    atr = self._calculate_atr(df)
                    price_range = round((df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1] * 100, 2)
                    
                    breakout_candidates[symbol] = {
                        'volume_ratio': volume_ratio,  # Now in percentage
                        'price_change': price_change,  # Now in percentage
                        'current_price': round(df['close'].iloc[-1], 2),
                        'current_volume': current_volume,
                        'avg_volume': round(avg_volume.iloc[-1], 2),
                        'atr': round(atr, 2),
                        'daily_range': price_range,  # Now in percentage
                        'breakout_direction': 'up' if price_change > 0 else 'down'
                    }
            
            return breakout_candidates
            
        except Exception as e:
            print(f"Error scanning for volume breakouts: {e}")
            return {}

    def _calculate_correlation_matrix(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate correlation matrix between different market components"""
        # Extract closing prices
        closes = pd.DataFrame({
            symbol: data[symbol]['close']
            for symbol in data.keys()
        })
        
        # Calculate correlation matrix
        return closes.corr().round(2)

    @staticmethod
    def _calculate_percentile(series: pd.Series, value: float) -> float:
        """Calculate the percentile of a value within a series"""
        return len(series[series <= value]) / len(series) * 100

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean().iloc[-1]
            
            return atr
            
        except Exception as e:
            print(f"Error calculating ATR: {e}")
            return 0.0

