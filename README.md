# MetaTrader 5 Integrated Reinforcement Learning Agent

A sophisticated reinforcement learning agent designed for MetaTrader 5 (MT5) integration, featuring real-time market monitoring, Sequential Monte Carlo (SMC) learning, automated backtesting, and intelligent trade execution.

## 🚀 Features

### 📡 Real-Time Market Monitoring
- **Live MT5 Integration**: Seamless connection to MetaTrader 5 platform
- **Multi-Symbol Monitoring**: Simultaneous tracking of multiple currency pairs
- **Multi-Timeframe Analysis**: Analysis across M1, M5, M15, H1, and higher timeframes
- **Real-Time Data Processing**: Continuous technical indicator calculation and pattern recognition

### 🧠 Advanced Reinforcement Learning
- **Sequential Monte Carlo (SMC)**: Particle-based learning for dynamic market adaptation
- **Multi-Agent Architecture**: Separate agents for different symbol-timeframe combinations
- **Continuous Learning**: Real-time strategy updates based on market feedback
- **Ensemble Decision Making**: Weighted particle voting for robust action selection

### 📊 Comprehensive Technical Analysis
- **50+ Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, ADX, and more
- **Candlestick Pattern Recognition**: Doji, Hammer, Engulfing patterns
- **Market Regime Detection**: Trending vs. ranging market identification
- **Volume Analysis**: Volume-based indicators and VWAP calculations

### 🧪 Advanced Backtesting Engine
- **Multi-Strategy Testing**: Compare different approaches across timeframes
- **Risk-Adjusted Metrics**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Drawdown Analysis**: Maximum drawdown and recovery time analysis
- **Monte Carlo Simulation**: Robust strategy validation

### ⚖️ Intelligent Risk Management
- **Position Sizing**: Kelly Criterion and volatility-based sizing
- **Dynamic Stop Losses**: ATR-based and trailing stop implementations
- **Correlation Management**: Avoid over-concentration in correlated pairs
- **Emergency Stop System**: Automatic trading halt on excessive drawdown

### 🔄 Strategy Optimization
- **Performance Monitoring**: Continuous strategy performance evaluation
- **Automatic Refactoring**: Strategy updates based on market regime changes
- **Particle Resampling**: SMC-based strategy evolution and improvement
- **Multi-Objective Optimization**: Balance return, risk, and consistency

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- MetaTrader 5 platform
- MT5 account (demo or live)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd ai_trading_bot

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir models reports logs data config
```

### MT5 Configuration
1. Install MetaTrader 5
2. Enable algorithmic trading in MT5 settings
3. Install the MetaTrader5 Python package
4. Ensure MT5 is running when using the agent

## 🚀 Usage

### Demo Mode (No MT5 Required)
```bash
python main.py --mode demo
```
Runs with simulated data to demonstrate capabilities.

### Training Mode
```bash
python main.py --mode train --symbols EURUSD GBPUSD --episodes 200
```
Train agents on historical MT5 data.

### Backtesting Mode
```bash
python main.py --mode backtest --symbols EURUSD GBPUSD USDJPY
```
Run comprehensive backtests on trained models.

### Live Trading Mode
```bash
# Monitor only (no automatic trading)
python main.py --mode live --symbols EURUSD GBPUSD

# Automatic trading (use with caution)
python main.py --mode live --symbols EURUSD --auto-trade
```

## 📁 Project Structure

```
mt5-rl-agent/
├── src/
│   ├── mt5_connector.py          # MT5 integration and data handling
│   ├── smc_reinforcement_learning.py  # SMC-based RL implementation
│   ├── technical_indicators.py   # Technical analysis tools
│   ├── trading_environment.py    # Gym-like trading environment
│   ├── backtesting_engine.py     # Comprehensive backtesting
│   ├── risk_management.py        # Risk management system
│   └── mt5_rl_agent.py          # Main agent orchestrator
├── main.py                       # Main execution script
├── requirements.txt              # Python dependencies
├── models/                       # Trained model storage
├── reports/                      # Backtest reports and plots
├── logs/                         # Application logs
├── data/                         # Historical data cache
└── config/                       # Configuration files
```

## 🔧 Configuration

### Agent Configuration
```python
# Example configuration
config = {
    'lookback_window': 50,           # Historical data window
    'training_frequency': 100,       # Steps between updates
    'confidence_threshold': 0.7,     # Minimum confidence for trading
    'max_positions_per_symbol': 1    # Position limits
}
```

### Risk Parameters
```python
risk_params = RiskParameters(
    max_position_size=0.1,          # 10% of equity per position
    max_daily_loss=0.02,            # 2% daily loss limit
    max_drawdown=0.15,              # 15% maximum drawdown
    stop_loss_pct=0.02,             # 2% stop loss
    take_profit_pct=0.04,           # 4% take profit
    risk_per_trade=0.01             # 1% risk per trade
)
```

## 📊 Performance Metrics

The system tracks comprehensive performance metrics:

- **Return Metrics**: Total return, annual return, monthly returns
- **Risk Metrics**: Sharpe ratio, Sortino ratio, maximum drawdown
- **Trading Metrics**: Win rate, profit factor, average trade duration
- **Risk-Adjusted Metrics**: Calmar ratio, risk-adjusted returns
- **Advanced Analytics**: Value at Risk (VaR), Expected Shortfall

## 🔬 Technical Details

### Sequential Monte Carlo Implementation
- **Particle Filter**: 50+ particles for strategy diversity
- **Resampling**: Systematic resampling based on performance
- **Mutation**: Network weight perturbation for exploration
- **Selection**: Fitness-based particle survival

### Neural Network Architecture
- **Policy Network**: Multi-layer perceptron with dropout
- **Value Network**: Separate value function estimation
- **Input Features**: 50+ technical indicators across timeframes
- **Output Actions**: Hold, Buy, Sell, Close Long, Close Short

### Real-Time Processing
- **Threading**: Separate threads for each symbol monitoring
- **Data Pipeline**: Efficient tick data to feature transformation
- **Memory Management**: Circular buffers for real-time data
- **Performance Optimization**: Vectorized calculations

## ⚠️ Risk Disclaimer

This software is for educational and research purposes. Trading financial instruments involves substantial risk and may not be suitable for all investors. Past performance does not guarantee future results. Always test thoroughly on demo accounts before live trading.

### Important Warnings
- **Start with Demo Trading**: Always test on demo accounts first
- **Risk Management**: Never risk more than you can afford to lose
- **Market Risk**: Market conditions can change rapidly
- **System Risk**: Technology failures can occur
- **Regulatory Risk**: Ensure compliance with local regulations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add comprehensive tests
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Review the documentation
- Check the examples in the demo mode

## 🔮 Future Enhancements

- **Deep Reinforcement Learning**: Integration with PPO, A3C algorithms
- **Market Sentiment Analysis**: News and social media sentiment integration
- **Multi-Asset Support**: Stocks, commodities, cryptocurrencies
- **Advanced Risk Models**: VaR models, stress testing
- **Web Interface**: Real-time monitoring dashboard
- **Cloud Deployment**: Scalable cloud-based execution