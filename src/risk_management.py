"""
Risk Management Module
Implements comprehensive risk management for trading operations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

class RiskLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    EXTREME = 4

@dataclass
class RiskParameters:
    max_position_size: float = 0.1  # Maximum position size as fraction of equity
    max_daily_loss: float = 0.02    # Maximum daily loss as fraction of equity
    max_drawdown: float = 0.15      # Maximum drawdown before stopping
    stop_loss_pct: float = 0.02     # Default stop loss percentage
    take_profit_pct: float = 0.04   # Default take profit percentage
    max_open_positions: int = 3     # Maximum number of open positions
    risk_per_trade: float = 0.01    # Risk per trade as fraction of equity
    correlation_threshold: float = 0.7  # Maximum correlation between positions

class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, initial_balance: float, risk_params: RiskParameters = None):
        self.initial_balance = initial_balance
        self.risk_params = risk_params or RiskParameters()
        
        # Risk tracking
        self.daily_pnl = 0.0
        self.current_drawdown = 0.0
        self.peak_equity = initial_balance
        self.open_positions = []
        self.trade_history = []
        
        # Risk alerts
        self.risk_alerts = []
        self.emergency_stop = False
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def evaluate_trade_risk(self, 
                           symbol: str,
                           action: str,
                           price: float,
                           current_equity: float,
                           market_data: pd.DataFrame = None) -> Dict:
        """Evaluate risk for a potential trade"""
        
        risk_assessment = {
            'approved': True,
            'risk_level': RiskLevel.LOW,
            'warnings': [],
            'position_size': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'risk_reward_ratio': 0.0
        }
        
        # Check if emergency stop is active
        if self.emergency_stop:
            risk_assessment['approved'] = False
            risk_assessment['warnings'].append("Emergency stop active")
            return risk_assessment
        
        # Check maximum positions limit
        if len(self.open_positions) >= self.risk_params.max_open_positions:
            risk_assessment['approved'] = False
            risk_assessment['warnings'].append("Maximum open positions reached")
            return risk_assessment
        
        # Check daily loss limit
        if self.daily_pnl <= -self.risk_params.max_daily_loss * current_equity:
            risk_assessment['approved'] = False
            risk_assessment['warnings'].append("Daily loss limit exceeded")
            return risk_assessment
        
        # Check drawdown limit
        if self.current_drawdown >= self.risk_params.max_drawdown:
            risk_assessment['approved'] = False
            risk_assessment['warnings'].append("Maximum drawdown exceeded")
            return risk_assessment
        
        # Calculate position size based on risk
        position_size = self._calculate_position_size(current_equity, price)
        risk_assessment['position_size'] = position_size
        
        # Calculate stop loss and take profit
        if action.upper() == 'BUY':
            stop_loss = price * (1 - self.risk_params.stop_loss_pct)
            take_profit = price * (1 + self.risk_params.take_profit_pct)
        else:  # SELL
            stop_loss = price * (1 + self.risk_params.stop_loss_pct)
            take_profit = price * (1 - self.risk_params.take_profit_pct)
        
        risk_assessment['stop_loss'] = stop_loss
        risk_assessment['take_profit'] = take_profit
        
        # Calculate risk-reward ratio
        risk_amount = abs(price - stop_loss) * position_size
        reward_amount = abs(take_profit - price) * position_size
        risk_assessment['risk_reward_ratio'] = reward_amount / max(risk_amount, 1e-8)
        
        # Check minimum risk-reward ratio
        if risk_assessment['risk_reward_ratio'] < 1.5:
            risk_assessment['warnings'].append("Low risk-reward ratio")
            risk_assessment['risk_level'] = RiskLevel.MEDIUM
        
        # Market volatility assessment
        if market_data is not None:
            volatility_risk = self._assess_volatility_risk(market_data)
            if volatility_risk > 0.5:
                risk_assessment['risk_level'] = RiskLevel.HIGH
                risk_assessment['warnings'].append("High market volatility")
        
        # Correlation risk check
        correlation_risk = self._check_correlation_risk(symbol)
        if correlation_risk:
            risk_assessment['warnings'].append("High correlation with existing positions")
            risk_assessment['risk_level'] = RiskLevel.MEDIUM
        
        return risk_assessment
    
    def _calculate_position_size(self, current_equity: float, price: float) -> float:
        """Calculate optimal position size based on risk parameters"""
        
        # Method 1: Fixed fractional method
        max_position_value = current_equity * self.risk_params.max_position_size
        
        # Method 2: Risk-based sizing
        risk_amount = current_equity * self.risk_params.risk_per_trade
        stop_loss_distance = price * self.risk_params.stop_loss_pct
        risk_based_size = risk_amount / (stop_loss_distance * 100000)  # Assuming forex
        
        # Use the smaller of the two
        position_size = min(
            max_position_value / (price * 100000),
            risk_based_size,
            self.risk_params.max_position_size
        )
        
        return max(0.01, position_size)  # Minimum position size
    
    def _assess_volatility_risk(self, market_data: pd.DataFrame) -> float:
        """Assess volatility risk based on recent market data"""
        if len(market_data) < 20:
            return 0.5  # Default medium risk
        
        # Calculate recent volatility
        returns = market_data['close'].pct_change().dropna()
        recent_vol = returns.tail(20).std()
        
        # Calculate historical volatility percentile
        historical_vol = returns.rolling(100).std().dropna()
        if len(historical_vol) > 0:
            vol_percentile = (recent_vol > historical_vol).mean()
            return vol_percentile
        
        return 0.5
    
    def _check_correlation_risk(self, symbol: str) -> bool:
        """Check if new position would create excessive correlation risk"""
        if len(self.open_positions) == 0:
            return False
        
        # Simplified correlation check based on currency pairs
        # In a real implementation, this would use actual correlation data
        existing_symbols = [pos['symbol'] for pos in self.open_positions]
        
        # Check for same base or quote currency
        if symbol in existing_symbols:
            return True
        
        # Check for highly correlated pairs (simplified)
        correlation_groups = [
            ['EURUSD', 'GBPUSD', 'AUDUSD'],  # USD pairs
            ['EURJPY', 'GBPJPY', 'AUDJPY'],  # JPY pairs
            ['EURGBP', 'EURAUD', 'EURCAD'],  # EUR crosses
        ]
        
        for group in correlation_groups:
            if symbol in group:
                for existing_symbol in existing_symbols:
                    if existing_symbol in group:
                        return True
        
        return False
    
    def update_position(self, position_data: Dict):
        """Update an existing position"""
        for i, pos in enumerate(self.open_positions):
            if pos['ticket'] == position_data['ticket']:
                self.open_positions[i].update(position_data)
                break
        else:
            # New position
            self.open_positions.append(position_data)
    
    def close_position(self, ticket: int, close_price: float, profit: float):
        """Record position closure"""
        for i, pos in enumerate(self.open_positions):
            if pos['ticket'] == ticket:
                closed_position = self.open_positions.pop(i)
                
                # Record trade
                trade_record = {
                    'ticket': ticket,
                    'symbol': closed_position['symbol'],
                    'open_price': closed_position['price_open'],
                    'close_price': close_price,
                    'profit': profit,
                    'duration': None,  # Calculate if timestamp available
                    'type': closed_position['type']
                }
                self.trade_history.append(trade_record)
                
                # Update daily P&L
                self.daily_pnl += profit
                
                self.logger.info(f"Position {ticket} closed with profit: ${profit:.2f}")
                break
    
    def update_equity(self, current_equity: float):
        """Update current equity and risk metrics"""
        # Update peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Update drawdown
        self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
        # Check for risk alerts
        self._check_risk_alerts(current_equity)
        
        # Emergency stop check
        if self.current_drawdown >= self.risk_params.max_drawdown:
            self.emergency_stop = True
            self.logger.warning("EMERGENCY STOP ACTIVATED - Maximum drawdown exceeded")
    
    def _check_risk_alerts(self, current_equity: float):
        """Check for various risk alert conditions"""
        alerts = []
        
        # Drawdown alerts
        if self.current_drawdown >= 0.10:
            alerts.append(f"High drawdown warning: {self.current_drawdown*100:.1f}%")
        
        # Daily loss alerts
        daily_loss_pct = abs(self.daily_pnl) / current_equity
        if daily_loss_pct >= 0.015:  # 1.5% daily loss
            alerts.append(f"High daily loss: {daily_loss_pct*100:.1f}%")
        
        # Concentration risk
        if len(self.open_positions) >= self.risk_params.max_open_positions - 1:
            alerts.append("Approaching maximum position limit")
        
        # Add new alerts
        for alert in alerts:
            if alert not in self.risk_alerts:
                self.risk_alerts.append(alert)
                self.logger.warning(f"RISK ALERT: {alert}")
    
    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk summary"""
        total_exposure = sum(
            abs(pos.get('volume', 0)) * pos.get('price_current', 0) * 100000 
            for pos in self.open_positions
        )
        
        unrealized_pnl = sum(pos.get('profit', 0) for pos in self.open_positions)
        
        return {
            'current_drawdown': self.current_drawdown,
            'daily_pnl': self.daily_pnl,
            'open_positions': len(self.open_positions),
            'total_exposure': total_exposure,
            'unrealized_pnl': unrealized_pnl,
            'risk_alerts': self.risk_alerts,
            'emergency_stop': self.emergency_stop,
            'risk_utilization': {
                'position_count': len(self.open_positions) / self.risk_params.max_open_positions,
                'daily_loss': abs(self.daily_pnl) / (self.initial_balance * self.risk_params.max_daily_loss),
                'drawdown': self.current_drawdown / self.risk_params.max_drawdown
            }
        }
    
    def reset_daily_pnl(self):
        """Reset daily P&L counter (call at start of new day)"""
        self.daily_pnl = 0.0
        self.logger.info("Daily P&L reset")
    
    def override_emergency_stop(self, reason: str):
        """Override emergency stop with reason (use carefully)"""
        self.emergency_stop = False
        self.risk_alerts.append(f"Emergency stop overridden: {reason}")
        self.logger.warning(f"Emergency stop overridden: {reason}")
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        if avg_loss == 0:
            return 0.0
        
        win_loss_ratio = abs(avg_win / avg_loss)
        kelly_pct = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Apply fractional Kelly to reduce risk
        fractional_kelly = kelly_pct * 0.25  # Use 25% of Kelly
        
        # Cap at maximum position size
        return min(max(0.01, fractional_kelly), self.risk_params.max_position_size)