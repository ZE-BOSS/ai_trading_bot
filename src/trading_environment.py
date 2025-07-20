"""
Trading Environment for Reinforcement Learning
Implements a Gym-like environment for MT5 trading
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from enum import Enum
import logging

class TradingAction(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE_BUY = 3
    CLOSE_SELL = 4

class TradingEnvironment(gym.Env):
    """Trading environment for RL agent"""

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.0001,
        max_position_size: float = 0.1,
        lookback_window: int = 50,
        reward_type: str = "profit"
    ):
        
        super().__init__()

        self.data = data.copy()
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.lookback_window = lookback_window
        self.reward_type = reward_type

        # State variables
        self.current_step = 0
        self.balance = initial_balance
        self.equity = initial_balance
        self.position = 0.0
        self.position_price = 0.0
        self.trade_history = []
        self.unrealized_pnl = 0.0

        # Action and observation spaces
        self.action_space = gym.spaces.Discrete(len(TradingAction))
        self.feature_columns = [col for col in self.data.columns if col not in ['time']]
        obs_size = len(self.feature_columns) * self.lookback_window + 4
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Metrics
        self.peak_equity = self.equity
        self.max_drawdown = 0.0
        self.total_trades = 0
        self.winning_trades = 0

        self.logger = logging.getLogger(__name__)

    def reset(self) -> Tuple[np.ndarray, Dict]:
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0
        self.equity = self.initial_balance
        self.trade_history = []
        self.done = False
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector"""
        if self.current_step < self.lookback_window:
            return np.zeros(self.lookback_window * len(self.data.columns))
        
        # Get features for lookback window
        features = []
        for i in range(self.current_step - self.lookback_window + 1, self.current_step + 1):
            if i < len(self.data):
                row = self.data.iloc[i]
                # Use only OHLC if volume doesn't exist
                if 'volume' in row:
                    features.extend([row['open'], row['high'], row['low'], row['close'], row['volume']])
                else:
                    # Fallback to OHLC only
                    features.extend([row['open'], row['high'], row['low'], row['close']])
        
        # Add account state
        features.extend([self.balance, self.position, self.equity, len(self.trade_history)])
        return features
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute trading action"""
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True
            return self._get_observation(), 0, True, False, {}
        
        reward = 0
        current_price = self.data.iloc[self.current_step]['close']
        
        # Action meanings: 0=hold, 1=buy, 2=sell, 3=close long, 4=close short
        if action == 1:  # Buy
            position_size = min(self.max_position_size * self.balance, self.balance / current_price)
            self.position += position_size
            self.balance -= position_size * current_price * (1 + self.transaction_cost)
        elif action == 2:  # Sell
            position_size = min(self.max_position_size * self.balance, self.balance / current_price)
            self.position -= position_size
            self.balance += position_size * current_price * (1 - self.transaction_cost)
        elif action == 3 and self.position > 0:  # Close long
            close_amount = min(self.position, self.position)
            self.balance += close_amount * current_price * (1 - self.transaction_cost)
            self.position -= close_amount
            reward = (current_price - self.trade_history[-1]['entry_price']) * close_amount
        elif action == 4 and self.position < 0:  # Close short
            close_amount = min(abs(self.position), abs(self.position))
            self.balance += close_amount * current_price * (1 - self.transaction_cost)
            self.position += close_amount
            reward = (self.trade_history[-1]['entry_price'] - current_price) * close_amount
        
        # Update equity
        prev_equity = self.equity
        self.equity = self.balance + self.position * current_price
        reward = self.equity - prev_equity  # P&L as reward
        
        # Record trade
        if action in [1, 2]:
            self.trade_history.append({
                'step': self.current_step,
                'action': action,
                'entry_price': current_price,
                'position': self.position
            })
        
        # Clip reward to prevent extreme values
        reward = max(min(reward, self.initial_balance), -self.initial_balance)
        
        return self._get_observation(), reward, self.done, False, {}
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        return {
            'total_return': (self.equity - self.initial_balance) / self.initial_balance,
            'sharpe_ratio': 0.0,  # Placeholder
            'max_drawdown': 0.0,   # Placeholder
            'win_rate': 0.0         # Placeholder
        }
    def _execute_action(self, action: int, current_price: float) -> float:
        action_enum = TradingAction(action)
        reward = 0.0
        profit = 0.0

        if action_enum == TradingAction.HOLD:
            reward = -0.001  # Small penalty to encourage action

        elif action_enum == TradingAction.BUY:
            if self.position <= 0:
                if self.position < 0:
                    profit = -self.position * (current_price - self.position_price) * 100000
                    self.balance += profit - abs(self.position) * current_price * self.transaction_cost * 100000
                    self._record_trade(profit)

                position_size = min(self.max_position_size, self.balance / (current_price * 100000))
                self.position = position_size
                self.position_price = current_price
                self.balance -= position_size * current_price * self.transaction_cost * 100000
                self.total_trades += 1
                reward = 0.01

        elif action_enum == TradingAction.SELL:
            if self.position >= 0:
                if self.position > 0:
                    profit = self.position * (current_price - self.position_price) * 100000
                    self.balance += profit - self.position * current_price * self.transaction_cost * 100000
                    self._record_trade(profit)

                position_size = min(self.max_position_size, self.balance / (current_price * 100000))
                self.position = -position_size
                self.position_price = current_price
                self.balance -= position_size * current_price * self.transaction_cost * 100000
                self.total_trades += 1
                reward = 0.01

        elif action_enum == TradingAction.CLOSE_BUY and self.position > 0:
            profit = self.position * (current_price - self.position_price) * 100000
            self.balance += profit - self.position * current_price * self.transaction_cost * 100000
            self._record_trade(profit)
            self.position = 0.0
            self.position_price = 0.0
            reward = self._calculate_reward(profit)

        elif action_enum == TradingAction.CLOSE_SELL and self.position < 0:
            profit = -self.position * (current_price - self.position_price) * 100000
            self.balance += profit - abs(self.position) * current_price * self.transaction_cost * 100000
            self._record_trade(profit)
            self.position = 0.0
            self.position_price = 0.0
            reward = self._calculate_reward(profit)

        return reward

    def _record_trade(self, profit: float):
        self.trade_history.append(profit)
        if profit > 0:
            self.winning_trades += 1

    def _calculate_reward(self, profit: float) -> float:
        if self.reward_type == "profit":
            return profit / 1000.0
        elif self.reward_type == "sharpe":
            returns = np.array(self.trade_history)
            if len(returns) < 2 or returns.std() == 0:
                return profit / 1000.0
            return returns.mean() / returns.std()
        elif self.reward_type == "risk_adjusted":
            base = profit / 1000.0
            penalty = -self.max_drawdown * 10
            consistency = (self.winning_trades / self.total_trades) * 2 if self.total_trades > 0 else 0
            return base + penalty + consistency
        return 0.0

    def _get_info(self) -> Dict:
        return {
            'balance': self.balance,
            'equity': self.equity,
            'position': self.position,
            'unrealized_pnl': self.unrealized_pnl,
            'total_trades': self.total_trades,
            'max_drawdown': self.max_drawdown
        }
