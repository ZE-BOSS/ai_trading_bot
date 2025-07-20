"""
Backtesting Engine
Comprehensive backtesting framework for strategy evaluation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from src.trading_environment import TradingEnvironment
from src.technical_indicators import TechnicalIndicators

class BacktestingEngine:
    """Comprehensive backtesting engine"""
    
    def __init__(self, 
                 initial_balance: float = 10000.0,
                 transaction_cost: float = 0.0001,
                 max_position_size: float = 0.1):
        
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        
        # Results storage
        self.results = []
        self.current_backtest = None
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def run_backtest(
        self, 
        agent, 
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        start_date: str = None,
        end_date: str = None
    ) -> Dict:
        """Run comprehensive backtest"""

        # Ensure volume exists
        if 'volume' not in data.columns:
            if 'tick_volume' in data.columns:
                data['volume'] = data['tick_volume']
            else:
                data['volume'] = 1000  # Default volume

        # Filter data by date range if specified
        if start_date or end_date:
            data = self._filter_data_by_date(data, start_date, end_date)
        
        # Prepare data with technical indicators
        processed_data = TechnicalIndicators.calculate_all_indicators(data)
        processed_data = TechnicalIndicators.detect_patterns(processed_data)
        processed_data = TechnicalIndicators.calculate_market_regime(processed_data)
        
        # Remove NaN values
        processed_data = processed_data.dropna()
        
        if len(processed_data) < 100:
            raise ValueError("Insufficient data for backtesting")
        
        # Create trading environment
        env = TradingEnvironment(
            data=processed_data,
            initial_balance=self.initial_balance,
            transaction_cost=self.transaction_cost,
            max_position_size=self.max_position_size
        )
        
        # Run backtest
        self.logger.info(f"Starting backtest for {symbol} on {timeframe}")
        start_time = datetime.now()
        
        obs, info = env.reset()
        done = False
        step_count = 0
        
        # Store detailed results
        step_results = []
        
        while not done:
            # Get action from agent
            action, action_info = agent.select_action(obs)
            
            # Execute step
            next_obs, reward, done, truncated, step_info = env.step(action)
            
            # Store step data
            step_data = {
                'step': step_count,
                'datetime': processed_data.index[env.current_step-1] if env.current_step > 0 else processed_data.index[0],
                'action': action,
                'reward': reward,
                'balance': step_info['balance'],
                'equity': step_info['equity'],
                'position': step_info['position'],
                'unrealized_pnl': step_info['unrealized_pnl'],
                'confidence': action_info.get('confidence', 0.0)
            }
            step_results.append(step_data)
            
            # Update agent (if in training mode)
            if hasattr(agent, 'update') and step_count > 0:
                agent.update(obs, action, reward, next_obs, done)
            
            obs = next_obs
            step_count += 1
            
            if step_count % 1000 == 0:
                self.logger.info(f"Processed {step_count} steps")
        
        end_time = datetime.now()
        
        # Calculate performance metrics
        performance_metrics = env.get_performance_metrics()
        
        # Additional analysis
        step_df = pd.DataFrame(step_results)
        
        # Calculate returns
        step_df['returns'] = step_df['equity'].pct_change()
        step_df['cumulative_returns'] = (step_df['equity'] / self.initial_balance - 1) * 100
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(step_df)
        
        # Trade analysis
        trade_analysis = self._analyze_trades(env.trade_history, step_df)
        
        # Compile results
        backtest_results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'start_date': processed_data.index[0],
            'end_date': processed_data.index[-1],
            'duration': end_time - start_time,
            'total_steps': step_count,
            'performance_metrics': performance_metrics,
            'risk_metrics': risk_metrics,
            'trade_analysis': trade_analysis,
            'step_data': step_df,
            'final_equity': step_df['equity'].iloc[-1],
            'max_equity': step_df['equity'].max(),
            'min_equity': step_df['equity'].min(),
        }
        
        self.current_backtest = backtest_results
        self.results.append(backtest_results)
        
        self.logger.info(f"Backtest completed in {end_time - start_time}")
        self.logger.info(f"Final equity: ${backtest_results['final_equity']:.2f}")
        self.logger.info(f"Total return: {performance_metrics.get('total_return', 0)*100:.2f}%")
        
        return backtest_results
    
    def _filter_data_by_date(self, data: pd.DataFrame, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Filter data by date range"""
        filtered_data = data.copy()
        
        if start_date:
            start_date = pd.to_datetime(start_date)
            filtered_data = filtered_data[filtered_data.index >= start_date]
            
        if end_date:
            end_date = pd.to_datetime(end_date)
            filtered_data = filtered_data[filtered_data.index <= end_date]
            
        return filtered_data
    
    def _calculate_risk_metrics(self, step_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive risk metrics"""
        returns = step_df['returns'].dropna()
        equity_curve = step_df['equity']
        
        if len(returns) == 0:
            return {}
        
        # Basic statistics
        annual_returns = returns.mean() * 252 * 24  # Assuming hourly data
        volatility = returns.std() * np.sqrt(252 * 24)
        
        # Sharpe ratio
        sharpe_ratio = annual_returns / max(volatility, 1e-8)
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252 * 24)
        sortino_ratio = annual_returns / max(downside_deviation, 1e-8)
        
        # Maximum drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annual_returns / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Value at Risk (VaR)
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        
        # Expected Shortfall (Conditional VaR)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Skewness and Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        return {
            'annual_return': annual_returns,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    
    def _analyze_trades(self, trade_history: List[float], step_df: pd.DataFrame) -> Dict:
        """Analyze trade patterns and statistics"""
        if not trade_history:
            return {}
        
        trades = np.array(trade_history)
        
        # Basic trade statistics
        winning_trades = trades[trades > 0]
        losing_trades = trades[trades < 0]
        
        analysis = {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades) if len(trades) > 0 else 0,
            'average_win': winning_trades.mean() if len(winning_trades) > 0 else 0,
            'average_loss': losing_trades.mean() if len(losing_trades) > 0 else 0,
            'largest_win': winning_trades.max() if len(winning_trades) > 0 else 0,
            'largest_loss': losing_trades.min() if len(losing_trades) > 0 else 0,
            'profit_factor': abs(winning_trades.sum()) / abs(losing_trades.sum()) if len(losing_trades) > 0 else float('inf'),
            'average_trade': trades.mean(),
            'total_profit': trades.sum()
        }
        
        # Consecutive trades analysis
        consecutive_wins = self._calculate_consecutive_runs(trades > 0)
        consecutive_losses = self._calculate_consecutive_runs(trades < 0)
        
        analysis.update({
            'max_consecutive_wins': max(consecutive_wins) if consecutive_wins else 0,
            'max_consecutive_losses': max(consecutive_losses) if consecutive_losses else 0,
            'avg_consecutive_wins': np.mean(consecutive_wins) if consecutive_wins else 0,
            'avg_consecutive_losses': np.mean(consecutive_losses) if consecutive_losses else 0
        })
        
        return analysis
    
    def _calculate_consecutive_runs(self, boolean_series: np.ndarray) -> List[int]:
        """Calculate consecutive runs in boolean series"""
        runs = []
        current_run = 0
        
        for value in boolean_series:
            if value:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                    current_run = 0
        
        if current_run > 0:
            runs.append(current_run)
            
        return runs
    
    def generate_report(self, save_path: str = None) -> str:
        """Generate comprehensive backtest report"""
        if not self.current_backtest:
            raise ValueError("No backtest results available")
        
        results = self.current_backtest
        
        # Generate report text
        report = f"""
=== BACKTESTING REPORT ===

Symbol: {results['symbol']}
Timeframe: {results['timeframe']}
Period: {results['start_date']} to {results['end_date']}
Duration: {results['duration']}

=== PERFORMANCE METRICS ===
Initial Balance: ${self.initial_balance:,.2f}
Final Equity: ${results['final_equity']:,.2f}
Total Return: {results['performance_metrics'].get('total_return', 0)*100:.2f}%
Max Drawdown: {results['risk_metrics'].get('max_drawdown', 0)*100:.2f}%

=== RISK METRICS ===
Sharpe Ratio: {results['risk_metrics'].get('sharpe_ratio', 0):.3f}
Sortino Ratio: {results['risk_metrics'].get('sortino_ratio', 0):.3f}
Calmar Ratio: {results['risk_metrics'].get('calmar_ratio', 0):.3f}
Annual Volatility: {results['risk_metrics'].get('volatility', 0)*100:.2f}%

=== TRADING STATISTICS ===
Total Trades: {results['trade_analysis'].get('total_trades', 0)}
Win Rate: {results['trade_analysis'].get('win_rate', 0)*100:.1f}%
Profit Factor: {results['trade_analysis'].get('profit_factor', 0):.2f}
Average Trade: ${results['trade_analysis'].get('average_trade', 0):.2f}
Largest Win: ${results['trade_analysis'].get('largest_win', 0):.2f}
Largest Loss: ${results['trade_analysis'].get('largest_loss', 0):.2f}

=== RISK ANALYSIS ===
VaR (95%): {results['risk_metrics'].get('var_95', 0)*100:.2f}%
VaR (99%): {results['risk_metrics'].get('var_99', 0)*100:.2f}%
CVaR (95%): {results['risk_metrics'].get('cvar_95', 0)*100:.2f}%
CVaR (99%): {results['risk_metrics'].get('cvar_99', 0)*100:.2f}%

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Report saved to {save_path}")
        
        return report
    
    def plot_results(self, save_path: str = None, show_plot: bool = True):
        """Generate comprehensive visualization of backtest results"""
        if not self.current_backtest:
            raise ValueError("No backtest results available")
        
        step_df = self.current_backtest['step_data']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Backtest Results - {self.current_backtest['symbol']}", fontsize=16)
        
        # Equity curve
        axes[0, 0].plot(step_df.index, step_df['equity'])
        axes[0, 0].axhline(y=self.initial_balance, color='r', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_ylabel('Equity ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Drawdown
        peak = step_df['equity'].expanding().max()
        drawdown = (step_df['equity'] - peak) / peak * 100
        axes[0, 1].fill_between(step_df.index, drawdown, 0, alpha=0.7, color='red')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Returns distribution
        returns = step_df['returns'].dropna()
        axes[1, 0].hist(returns, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
        axes[1, 0].set_title('Returns Distribution')
        axes[1, 0].set_xlabel('Returns')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Position over time
        axes[1, 1].plot(step_df.index, step_df['position'])
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 1].set_title('Position Size Over Time')
        axes[1, 1].set_ylabel('Position Size')
        axes[1, 1].set_xlabel('Time Steps')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def compare_strategies(self, results_list: List[Dict], save_path: str = None) -> pd.DataFrame:
        """Compare multiple backtest results"""
        comparison_data = []
        
        for i, results in enumerate(results_list):
            data = {
                'Strategy': f"Strategy_{i+1}",
                'Symbol': results['symbol'],
                'Total_Return': results['performance_metrics'].get('total_return', 0) * 100,
                'Sharpe_Ratio': results['risk_metrics'].get('sharpe_ratio', 0),
                'Max_Drawdown': results['risk_metrics'].get('max_drawdown', 0) * 100,
                'Win_Rate': results['trade_analysis'].get('win_rate', 0) * 100,
                'Total_Trades': results['trade_analysis'].get('total_trades', 0),
                'Profit_Factor': results['trade_analysis'].get('profit_factor', 0),
                'Final_Equity': results['final_equity']
            }
            comparison_data.append(data)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if save_path:
            comparison_df.to_csv(save_path, index=False)
            self.logger.info(f"Comparison saved to {save_path}")
        
        return comparison_df