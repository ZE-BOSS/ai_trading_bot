"""
Main MT5 Reinforcement Learning Agent
Integrates all components for autonomous trading
"""

import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
import os
import numpy as np
import pandas as pd

from src.mt5_connector import MT5Connector
from src.smc_reinforcement_learning import SMCReinforcementLearning
from src.technical_indicators import TechnicalIndicators
from src.trading_environment import TradingEnvironment
from src.backtesting_engine import BacktestingEngine
from src.risk_management import RiskManager, RiskParameters

class MT5RLAgent:
    """MetaTrader 5 Integrated Reinforcement Learning Agent"""
    
    def __init__(
        self, 
        mt5_login: int = None,
        mt5_password: str = None,
        mt5_server: str = None,
        initial_balance: float = 10000.0,
        symbols: List[str] = None,
        timeframes: List[int] = None
    ):
        
        # Initialize components
        self.mt5 = MT5Connector(mt5_login, mt5_password, mt5_server)
        self.symbols = symbols or ['EURUSD', 'GBPUSD', 'USDJPY']
        self.timeframes = timeframes or [1, 5, 15, 60]  # M1, M5, M15, H1
        self.initial_balance = initial_balance
        
        # Risk management
        self.risk_manager = RiskManager(initial_balance)
        
        # RL components
        self.agents = {}  # One agent per symbol-timeframe combination
        self.training_environments = {}
        
        # Data storage
        self.market_data = {}
        self.feature_data = {}
        
        # Control flags
        self.is_training = False
        self.is_live_trading = False
        self.auto_trade = False
        
        # Performance tracking
        self.performance_log = []
        self.trade_log = []
        
        # Threading
        self.threads = {}
        self.stop_event = threading.Event()
        
        # Backtesting
        self.backtesting_engine = BacktestingEngine(initial_balance)
        
        # Configuration
        self.config = {
            'lookback_window': 50,
            'training_frequency': 100,  # Steps between training updates
            'model_save_frequency': 1000,  # Steps between model saves
            'confidence_threshold': 0.7,  # Minimum confidence for live trading
            'max_positions_per_symbol': 1
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('mt5_rl_agent.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize state dimensions (will be calculated after data preparation)
        self.state_dim = None
        self.action_dim = 5  # HOLD, BUY, SELL, CLOSE_BUY, CLOSE_SELL
        
    def connect_mt5(self) -> bool:
        """Connect to MetaTrader 5"""
        success = self.mt5.connect()
        if success:
            self.logger.info("Successfully connected to MT5")
            # Start monitoring
            self.mt5.start_monitoring(self.symbols, self._on_tick_received)
        else:
            self.logger.error("Failed to connect to MT5")
        return success
    
    def disconnect_mt5(self):
        """Disconnect from MT5"""
        self.stop_all_operations()
        self.mt5.disconnect()
        self.logger.info("Disconnected from MT5")
    
    def _on_tick_received(self, tick_data: Dict):
        """Handle incoming tick data"""
        symbol = tick_data['symbol']
        
        # Update market data
        if symbol not in self.market_data:
            self.market_data[symbol] = []
        
        # Store tick data (you might want to aggregate to OHLC)
        self.market_data[symbol].append(tick_data)
        
        # Keep only recent data
        if len(self.market_data[symbol]) > 10000:
            self.market_data[symbol] = self.market_data[symbol][-5000:]
        
        # Trigger live trading decision if enabled
        if self.is_live_trading and self.auto_trade:
            self._make_trading_decision(symbol, tick_data)
    
    def prepare_training_data(self, symbol: str, timeframe: int, count: int = 5000) -> pd.DataFrame:
        """Prepare and process training data"""
        # Get historical data
        raw_data = self.mt5.get_historical_data(symbol, timeframe, count)

        # Check if data is empty
        if raw_data.empty:
            raise ValueError(f"No data available for {symbol} on timeframe {timeframe}")
        
        # Convert DataFrame to dictionary format for TechnicalIndicators
        data_dict = {
            'open': raw_data['open'].tolist(),
            'high': raw_data['high'].tolist(),
            'low': raw_data['low'].tolist(),
            'close': raw_data['close'].tolist(),
            'volume': raw_data['volume'].tolist()
        }
        
        # Calculate technical indicators
        processed_data = TechnicalIndicators.calculate_all_indicators(data_dict)

        # Convert back to DataFrame
        processed_data = pd.DataFrame(processed_data)

        # Remove NaN values
        processed_data = processed_data.dropna()
        
        # Store feature data
        key = f"{symbol}_{timeframe}"
        self.feature_data[key] = processed_data
        
        self.logger.info(f"Prepared {len(processed_data)} data points for {symbol} on timeframe {timeframe}")
        
        # Scale features to [0, 1] range
        for column in processed_data.columns:
            if column != 'time':
                col_min = processed_data[column].min()
                col_max = processed_data[column].max()
                if col_max - col_min > 0:
                    processed_data[column] = (processed_data[column] - col_min) / (col_max - col_min)
        
        return processed_data
    
    def initialize_agents(self):
        """Initialize RL agents for each symbol-timeframe combination"""
        if not self.feature_data:
            self.logger.error("No feature data available to initialize agents")
            return
    
        # Calculate state dimension from first dataset
        first_key = list(self.feature_data.keys())[0]
        sample_data = self.feature_data[first_key]
        
        # Calculate state dimension
        feature_columns = [col for col in sample_data.columns if col not in ['time']]
        self.state_dim = len(feature_columns) * self.config['lookback_window'] + 4
        
        self.logger.info(f"State dimension: {self.state_dim}")
        
        # Initialize agents
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                key = f"{symbol}_{timeframe}"
                
                if key in self.feature_data:
                    # Create RL agent
                    agent = SMCReinforcementLearning(
                        state_dim=self.state_dim,
                        action_dim=self.action_dim,
                        num_particles=50
                    )
                    
                    # Create training environment
                    env = TradingEnvironment(
                        data=self.feature_data[key],
                        initial_balance=self.initial_balance,
                        lookback_window=self.config['lookback_window']
                    )
                    
                    self.agents[key] = agent
                    self.training_environments[key] = env
                    
                    self.logger.info(f"Initialized agent for {key}")
    
    def train_agents(self, episodes: int = 100):
        """Train all RL agents"""
        if not self.agents:
            raise ValueError("No agents initialized. Call initialize_agents() first.")
        
        self.is_training = True
        self.logger.info(f"Starting training for {episodes} episodes")
        
        training_results = {}
        
        for key, agent in self.agents.items():
            self.logger.info(f"Training agent {key}")
            
            env = self.training_environments[key]
            episode_rewards = []
            
            for episode in range(episodes):
                obs, info = env.reset()
                done = False
                episode_reward = 0
                step_count = 0

                self.logger.info(f"Starting episode {episode + 1} for {key}")
                
                while not done:
                    # Get action from agent - returns tuple (action, action_info)
                    action, action_info = agent.select_action(obs)

                    self.logger.info(f"Episode {episode + 1}, Step {step_count + 1}, Action: {action}, Info: {action_info}")
                    
                    # Execute step
                    next_obs, reward, done, truncated, step_info = env.step(action)

                    self.logger.info(f"Episode {episode + 1}, Step {step_count + 1}, Reward: {reward}, Done: {done}")
                    
                    # Update agent - pass all parameters separately
                    agent.update(obs, action, reward, next_obs, done)

                    self.logger.info(f"Episode {episode + 1}, Step {step_count + 1}, Updated agent")
                    
                    obs = next_obs
                    episode_reward += reward
                    step_count += 1

                    self.logger.info(f"Episode {episode + 1}, Step {step_count + 1}, Total Reward: {episode_reward:.4f}")
                    
                    # Save model periodically
                    if step_count % self.config['model_save_frequency'] == 0:
                        model_path = f"models/{key}_step_{step_count}.pkl"
                        os.makedirs("models", exist_ok=True)
                        agent.save_model(model_path)
                        self.logger.info(f"Model saved at {model_path}")

                self.logger.info(f"Episode {episode + 1}/{episodes} completed for {key}, Reward: {episode_reward:.4f}")
                
                episode_rewards.append(episode_reward)
                
                if episode % 10 == 0:
                    avg_reward = np.mean(episode_rewards[-10:])
                    self.logger.info(f"{key} Episode {episode}, Avg Reward: {avg_reward:.4f}")
            
            training_results[key] = {
                'episode_rewards': episode_rewards,
                'final_performance': env.get_performance_metrics()
            }

            self.logger.info(f"Training completed for {key}, Final Performance: {training_results[key]['final_performance']}")
            
            # Save final model
            model_path = f"models/{key}_final.pkl"
            os.makedirs("models", exist_ok=True)
            agent.save_model(model_path)

            self.logger.info(f"Final model saved at {model_path}")
        
        self.is_training = False
        self.logger.info("Training completed")
        
        return training_results
    
    def run_backtests(self, start_date: str = None, end_date: str = None) -> Dict:
        """Run comprehensive backtests for all agents"""
        if not self.agents:
            raise ValueError("No agents initialized")
        
        backtest_results = {}
        
        for key, agent in self.agents.items():
            symbol, timeframe = key.split('_')
            timeframe = int(timeframe)
            
            self.logger.info(f"Running backtest for {key}")
            
            # Get data for backtesting
            data = self.feature_data[key]
            
            try:
                result = self.backtesting_engine.run_backtest(
                    agent=agent,
                    data=data,
                    symbol=symbol,
                    timeframe=str(timeframe),
                    start_date=start_date,
                    end_date=end_date
                )
                
                backtest_results[key] = result
                
                # Generate report
                report_path = f"reports/backtest_{key}.txt"
                os.makedirs("reports", exist_ok=True)
                self.backtesting_engine.generate_report(report_path)
                
                # Generate plots
                plot_path = f"reports/backtest_{key}.png"
                self.backtesting_engine.plot_results(plot_path, show_plot=False)
                
            except Exception as e:
                self.logger.error(f"Backtest failed for {key}: {e}")
                backtest_results[key] = None
        
        return backtest_results
    
    def start_live_trading(self, auto_trade: bool = False):
        """Start live trading mode"""
        if not self.mt5.connected:
            raise ConnectionError("Not connected to MT5")
        
        if not self.agents:
            raise ValueError("No agents initialized")
        
        self.is_live_trading = True
        self.auto_trade = auto_trade
        
        # Start monitoring threads for each symbol
        for symbol in self.symbols:
            thread = threading.Thread(
                target=self._live_trading_loop,
                args=(symbol,),
                name=f"LiveTrading_{symbol}"
            )
            thread.start()
            self.threads[f"live_{symbol}"] = thread
        
        self.logger.info(f"Live trading started (auto_trade: {auto_trade})")
    
    def stop_live_trading(self):
        """Stop live trading"""
        self.is_live_trading = False
        self.auto_trade = False
        
        # Wait for threads to stop
        for thread in self.threads.values():
            if thread.is_alive():
                thread.join(timeout=5)
        
        self.logger.info("Live trading stopped")
    
    def _live_trading_loop(self, symbol: str):
        """Live trading loop for a specific symbol"""
        while self.is_live_trading and not self.stop_event.is_set():
            try:
                # Check positions and market conditions
                self._monitor_positions(symbol)
                
                # Make trading decisions
                if self.auto_trade:
                    self._evaluate_trading_opportunities(symbol)
                
                time.sleep(1)  # 1 second loop
                
            except Exception as e:
                self.logger.error(f"Error in live trading loop for {symbol}: {e}")
                time.sleep(5)
    
    def _make_trading_decision(self, symbol: str, tick_data: Dict):
        """Make trading decision based on current market conditions"""
        try:
            # Get current positions
            positions = self.mt5.get_positions()
            symbol_positions = [pos for pos in positions if pos['symbol'] == symbol]
            
            # Check if we already have positions for this symbol
            if len(symbol_positions) >= self.config['max_positions_per_symbol']:
                return
            
            # Get the best performing agent for this symbol
            best_agent_key = self._get_best_agent_for_symbol(symbol)
            if not best_agent_key:
                return
            
            agent = self.agents[best_agent_key]
            
            # Prepare current state
            current_state = self._prepare_current_state(symbol, best_agent_key)
            if current_state is None:
                return
            
            # Get action from agent
            action, action_info = agent.select_action(current_state)
            confidence = action_info.get('confidence', 0.0)
            
            # Check confidence threshold
            if confidence < self.config['confidence_threshold']:
                return
            
            # Execute trade based on action
            self._execute_live_trade(symbol, action, tick_data, confidence)
            
        except Exception as e:
            self.logger.error(f"Error making trading decision for {symbol}: {e}")
    
    def _get_best_agent_for_symbol(self, symbol: str) -> Optional[str]:
        """Get the best performing agent for a symbol"""
        symbol_agents = [key for key in self.agents.keys() if key.startswith(symbol)]
        
        if not symbol_agents:
            return None
        
        # For now, return the H1 timeframe agent if available, otherwise the first one
        h1_key = f"{symbol}_60"
        if h1_key in symbol_agents:
            return h1_key
        
        return symbol_agents[0]
    
    def _prepare_current_state(self, symbol: str, agent_key: str) -> Optional[np.ndarray]:
        """Prepare current state for the agent"""
        try:
            # Get recent data
            timeframe = int(agent_key.split('_')[1])
            recent_data = self.mt5.get_historical_data(symbol, timeframe, self.config['lookback_window'] + 10)
            
            if len(recent_data) < self.config['lookback_window']:
                return None
            
            # Process data
            processed_data = TechnicalIndicators.calculate_all_indicators(recent_data)
            processed_data = TechnicalIndicators.detect_patterns(processed_data)
            processed_data = TechnicalIndicators.calculate_market_regime(processed_data)
            processed_data = processed_data.dropna()
            
            if len(processed_data) < self.config['lookback_window']:
                return None
            
            # Create mock environment to get state
            env = TradingEnvironment(
                data=processed_data,
                initial_balance=self.initial_balance,
                lookback_window=self.config['lookback_window']
            )
            
            # Reset to last state
            env.current_step = len(processed_data) - 1
            state = env._get_observation()
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error preparing state for {symbol}: {e}")
            return None
    
    def _execute_live_trade(self, symbol: str, action: int, tick_data: Dict, confidence: float):
        """Execute live trade based on agent decision"""
        try:
            current_price = tick_data['bid']  # Use bid for selling, ask for buying
            
            # Get risk assessment
            action_names = ['HOLD', 'BUY', 'SELL', 'CLOSE_BUY', 'CLOSE_SELL']
            action_name = action_names[action]
            
            if action_name in ['HOLD', 'CLOSE_BUY', 'CLOSE_SELL']:
                return  # Skip these actions for now
            
            # Get current equity
            account_info = self.mt5.get_positions()
            total_equity = sum(pos['profit'] for pos in account_info) + self.initial_balance
            
            # Risk assessment
            risk_assessment = self.risk_manager.evaluate_trade_risk(
                symbol=symbol,
                action=action_name,
                price=current_price,
                current_equity=total_equity
            )
            
            if not risk_assessment['approved']:
                self.logger.warning(f"Trade rejected for {symbol}: {risk_assessment['warnings']}")
                return
            
            # Execute trade
            if action_name == 'BUY':
                order_type = 0  # mt5.ORDER_TYPE_BUY
                price = tick_data['ask']
            else:  # SELL
                order_type = 1  # mt5.ORDER_TYPE_SELL
                price = tick_data['bid']
            
            volume = risk_assessment['position_size']
            stop_loss = risk_assessment['stop_loss']
            take_profit = risk_assessment['take_profit']
            
            result = self.mt5.place_order(
                symbol=symbol,
                order_type=order_type,
                volume=volume,
                price=price,
                sl=stop_loss,
                tp=take_profit,
                comment=f"RL_Agent_{action_name}_conf_{confidence:.2f}"
            )

            recent_trades = [t for t in self.trade_log if t['symbol'] == symbol][-10:]
            avg_confidence = (sum(t['confidence'] for t in recent_trades) / len(recent_trades)) if recent_trades else confidence
            
            if result['success']:
                trade_info = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': action_name,
                    'volume': volume,
                    'price': price,
                    'confidence': confidence,
                    'ticket': result['ticket'],
                    'avg_confidence': avg_confidence,
                    'take_profit': take_profit
                }
                
                self.trade_log.append(trade_info)
                self.logger.info(f"Trade executed: {trade_info}")
                
            else:
                self.logger.error(f"Trade failed: {result['error']}")
                
        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol}: {e}")
    
    def _monitor_positions(self, symbol: str):
        """Monitor and manage open positions"""
        try:
            positions = self.mt5.get_positions()
            symbol_positions = [pos for pos in positions if pos['symbol'] == symbol]
            
            for position in symbol_positions:
                # Update risk manager
                self.risk_manager.update_position(position)
                
                # Check for position management actions
                # (stop loss adjustments, trailing stops, etc.)
                self._manage_position(position)
                
        except Exception as e:
            self.logger.error(f"Error monitoring positions for {symbol}: {e}")
    
    def _manage_position(self, position: Dict):
        """Manage individual position (trailing stops, etc.)"""
        # Implement position management logic here
        # This could include:
        # - Trailing stop loss
        # - Partial profit taking
        # - Time-based exits
        # - Correlation-based exits
        pass
    
    def _evaluate_trading_opportunities(self, symbol: str):
        """Evaluate and potentially execute trading opportunities"""
        # This method can implement additional trading logic
        # such as:
        # - Multi-timeframe analysis
        # - Market regime filtering
        # - Economic calendar awareness
        # - Risk budget management
        pass
    
    def stop_all_operations(self):
        """Stop all operations and threads"""
        self.stop_event.set()
        self.is_training = False
        self.is_live_trading = False
        self.auto_trade = False
        
        # Stop all threads
        for thread_name, thread in self.threads.items():
            if thread.is_alive():
                self.logger.info(f"Stopping {thread_name}")
                thread.join(timeout=10)
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        # Get account information
        positions = self.mt5.get_positions()
        total_unrealized_pnl = sum(pos['profit'] for pos in positions)
        
        # Get risk summary
        risk_summary = self.risk_manager.get_risk_summary()
        
        # Calculate trading statistics
        if self.trade_log:
            recent_trades = self.trade_log[-100:]  # Last 100 trades
            trade_analysis = {
                'total_trades': len(self.trade_log),
                'recent_trades': len(recent_trades),
                'avg_confidence': np.mean([t['confidence'] for t in recent_trades]),
                'symbols_traded': len(set(t['symbol'] for t in recent_trades))
            }
        else:
            trade_analysis = {'total_trades': 0}
        
        return {
            'timestamp': datetime.now(),
            'account_status': {
                'initial_balance': self.initial_balance,
                'unrealized_pnl': total_unrealized_pnl,
                'open_positions': len(positions)
            },
            'risk_summary': risk_summary,
            'trade_analysis': trade_analysis,
            'agent_status': {
                'total_agents': len(self.agents),
                'training_active': self.is_training,
                'live_trading_active': self.is_live_trading,
                'auto_trade_enabled': self.auto_trade
            }
        }
    
    def save_configuration(self, filepath: str):
        """Save agent configuration"""
        config_data = {
            'symbols': self.symbols,
            'timeframes': self.timeframes,
            'initial_balance': self.initial_balance,
            'config': self.config,
            'risk_parameters': {
                'max_position_size': self.risk_manager.risk_params.max_position_size,
                'max_daily_loss': self.risk_manager.risk_params.max_daily_loss,
                'max_drawdown': self.risk_manager.risk_params.max_drawdown,
                'stop_loss_pct': self.risk_manager.risk_params.stop_loss_pct,
                'take_profit_pct': self.risk_manager.risk_params.take_profit_pct
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        self.logger.info(f"Configuration saved to {filepath}")
    
    def load_configuration(self, filepath: str):
        """Load agent configuration"""
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        
        self.symbols = config_data['symbols']
        self.timeframes = config_data['timeframes']
        self.initial_balance = config_data['initial_balance']
        self.config.update(config_data['config'])
        
        # Update risk parameters
        risk_params = config_data.get('risk_parameters', {})
        for key, value in risk_params.items():
            if hasattr(self.risk_manager.risk_params, key):
                setattr(self.risk_manager.risk_params, key, value)
        
        self.logger.info(f"Configuration loaded from {filepath}")