"""
Technical Indicators Calculator
Comprehensive technical analysis toolkit
"""
import math
from typing import List, Dict, Tuple, Optional
import numpy as np

class TechnicalIndicators:
    """Calculate various technical indicators"""
    
    @staticmethod
    def calculate_all_indicators(data: dict) -> dict:
        """Calculate all technical indicators for a dataset"""
        if not data or 'close' not in data:
            return {}
        
        # Create a new dictionary to store processed data
        processed_data = data.copy()
        
        # Calculate indicators
        processed_data['sma_20'] = TechnicalIndicators.sma(data['close'], 20)
        processed_data['sma_50'] = TechnicalIndicators.sma(data['close'], 50)
        processed_data['ema_12'] = TechnicalIndicators.ema(data['close'], 12)
        processed_data['ema_26'] = TechnicalIndicators.ema(data['close'], 26)
        processed_data['rsi_14'] = TechnicalIndicators.rsi(data['close'], 14)
        
        # Add Bollinger Bands
        bbands = TechnicalIndicators.bollinger_bands(data['close'], 20, 2)
        processed_data.update(bbands)
        
        # Add MACD
        macd_data = TechnicalIndicators.macd(data['close'])
        processed_data.update(macd_data)
        
        # Add pattern detection
        patterns = TechnicalIndicators.detect_patterns(data)
        processed_data.update(patterns)
        
        # Add market regime
        market_regime = TechnicalIndicators.calculate_market_regime(data)
        processed_data.update(market_regime)
        
        return processed_data
    
    @staticmethod
    def detect_patterns(data: dict) -> dict:
        """Detect candlestick patterns (simplified implementation)"""
        patterns = {
            'is_doji': [0] * len(data['close']),
            'is_hammer': [0] * len(data['close']),
            'is_engulfing': [0] * len(data['close'])
        }
        
        # Simple pattern detection logic
        for i in range(1, len(data['close'])):
            open_price = data['open'][i]
            close_price = data['close'][i]
            high_price = data['high'][i]
            low_price = data['low'][i]
            prev_close = data['close'][i-1]
            prev_open = data['open'][i-1]
            
            # Doji pattern (small body)
            body_size = abs(open_price - close_price)
            candle_range = high_price - low_price
            if candle_range > 0 and body_size / candle_range < 0.1:
                patterns['is_doji'][i] = 1
                
            # Hammer pattern (long lower wick)
            lower_wick = min(open_price, close_price) - low_price
            if lower_wick > 0 and body_size > 0:
                if lower_wick > 2 * body_size and (high_price - max(open_price, close_price)) < body_size:
                    patterns['is_hammer'][i] = 1
                    
            # Engulfing pattern
            if (prev_close < prev_open and  # Previous candle is bearish
                close_price > open_price and  # Current candle is bullish
                open_price < prev_close and 
                close_price > prev_open):
                patterns['is_engulfing'][i] = 1
        
        return patterns
    
    @staticmethod
    def calculate_market_regime(data: dict) -> dict:
        """Calculate market regime (simplified implementation)"""
        # 0 = ranging, 1 = uptrend, 2 = downtrend
        market_regime = [0] * len(data['close'])
        sma_20 = TechnicalIndicators.sma(data['close'], 20)
        sma_50 = TechnicalIndicators.sma(data['close'], 50)
        
        for i in range(50, len(data['close'])):
            if sma_20[i] is None or sma_50[i] is None:
                continue
                
            # Uptrend: short MA above long MA
            if sma_20[i] > sma_50[i]:
                market_regime[i] = 1
            # Downtrend: short MA below long MA
            elif sma_20[i] < sma_50[i]:
                market_regime[i] = 2
        
        return {'market_regime': market_regime}
    
    @staticmethod
    def sma(data: List[float], period: int) -> List[Optional[float]]:
        """Simple Moving Average"""
        result = []
        for i in range(len(data)):
            if i < period - 1:
                result.append(None)
            else:
                avg = sum(data[i-period+1:i+1]) / period
                result.append(avg)
        return result
    
    @staticmethod
    def ema(data: List[float], period: int) -> List[Optional[float]]:
        """Exponential Moving Average"""
        if not data:
            return []
        
        result = [None] * (period - 1)
        if len(data) >= period:
            # First EMA value is SMA
            sma_val = sum(data[:period]) / period
            result.append(sma_val)
            
            # Calculate multiplier
            multiplier = 2 / (period + 1)
            
            # Calculate subsequent EMA values
            for i in range(period, len(data)):
                ema_val = (data[i] * multiplier) + (result[-1] * (1 - multiplier))
                result.append(ema_val)
        
        return result
    
    @staticmethod
    def rsi(data: List[float], period: int = 14) -> List[Optional[float]]:
        """Relative Strength Index"""
        if len(data) < period + 1:
            return [None] * len(data)
        
        # Calculate price changes
        deltas = []
        for i in range(1, len(data)):
            deltas.append(data[i] - data[i-1])
        
        result = [None] * (period)
        
        # Calculate initial average gain and loss
        gains = [max(d, 0) for d in deltas[:period]]
        losses = [abs(min(d, 0)) for d in deltas[:period]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            result.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi_val = 100 - (100 / (1 + rs))
            result.append(rsi_val)
        
        # Calculate subsequent RSI values
        for i in range(period, len(deltas)):
            gain = max(deltas[i], 0)
            loss = abs(min(deltas[i], 0))
            
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
            
            if avg_loss == 0:
                result.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi_val = 100 - (100 / (1 + rs))
                result.append(rsi_val)
        
        return result
    
    @staticmethod
    def bollinger_bands(data: List[float], period: int = 20, std_dev: int = 2) -> Dict[str, List[float]]:
        """Bollinger Bands"""
        sma_values = TechnicalIndicators.sma(data, period)
        
        # Calculate standard deviation manually
        std_values = []
        for i in range(len(data)):
            if i < period - 1:
                std_values.append(None)
            else:
                window_data = data[i-period+1:i+1]
                mean_val = sum(window_data) / period
                variance = sum((x - mean_val) ** 2 for x in window_data) / period
                std_val = math.sqrt(variance)
                std_values.append(std_val)
        
        upper_band = []
        lower_band = []
        
        for i in range(len(data)):
            if sma_values[i] is None or std_values[i] is None:
                upper_band.append(None)
                lower_band.append(None)
            else:
                upper_band.append(sma_values[i] + (std_dev * std_values[i]))
                lower_band.append(sma_values[i] - (std_dev * std_values[i]))
        
        return {
            'upper': upper_band,
            'middle': sma_values,
            'lower': lower_band
        }
    
    @staticmethod
    def macd(data: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, List[float]]:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        
        # Calculate MACD line
        macd_line = []
        for i in range(len(data)):
            if ema_fast[i] is None or ema_slow[i] is None:
                macd_line.append(None)
            else:
                macd_line.append(ema_fast[i] - ema_slow[i])
        
        # Calculate signal line (EMA of MACD line)
        # Filter out None values for signal calculation
        macd_values = [x for x in macd_line if x is not None]
        if macd_values:
            signal_ema = TechnicalIndicators.ema(macd_values, signal)
            # Pad with None values to match original length
            none_count = len([x for x in macd_line if x is None])
            signal_line = [None] * none_count + signal_ema
        else:
            signal_line = [None] * len(macd_line)
        
        # Calculate histogram
        histogram = []
        for i in range(len(macd_line)):
            if macd_line[i] is None or i < len(signal_line) and signal_line[i] is None:
                histogram.append(None)
            else:
                hist_val = macd_line[i] - (signal_line[i] if i < len(signal_line) else 0)
                histogram.append(hist_val)
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }