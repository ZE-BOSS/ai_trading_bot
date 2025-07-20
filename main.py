"""
Main execution script for MT5 RL Agent
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.mt5_rl_agent import MT5RLAgent

def setup_directories():
    """Create necessary directories"""
    directories = ['models', 'reports', 'logs', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='MT5 Reinforcement Learning Agent')
    parser.add_argument('--mode', choices=['train', 'backtest', 'live', 'demo'], 
                       default='demo', help='Operation mode')
    parser.add_argument('--symbols', nargs='+', default=['EURUSD', 'GBPUSD'], 
                       help='Trading symbols')
    parser.add_argument('--episodes', type=int, default=100, 
                       help='Training episodes')
    parser.add_argument('--auto-trade', action='store_true', 
                       help='Enable automatic trading in live mode')
    parser.add_argument('--config', type=str, help='Configuration file path')
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Initialize agent
    agent = MT5RLAgent(
        mt5_login=210526788,
        mt5_password='S@jasper&12345',
        mt5_server='Exness-MT5Trial9',
        initial_balance=10000.0,
        symbols=args.symbols,
        timeframes=[1, 5, 15, 60],  # M1, M5, M15, H1
    )
    
    # Load configuration if provided
    if args.config and os.path.exists(args.config):
        agent.load_configuration(args.config)
    
    try:
        if args.mode == 'demo':
            run_demo_mode(agent)
        elif args.mode == 'train':
            run_training_mode(agent, args.episodes)
        elif args.mode == 'backtest':
            run_backtest_mode(agent)
        elif args.mode == 'live':
            run_live_mode(agent, args.auto_trade)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        agent.stop_all_operations()
        agent.disconnect_mt5()
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        agent.stop_all_operations()
        agent.disconnect_mt5()

def run_demo_mode(agent: MT5RLAgent):
    """Run in demo mode with simulated data"""
    print("=== MT5 RL Agent Demo Mode ===")
    print("This mode demonstrates the agent capabilities using simulated data")
    
    # Create sample data
    import pandas as pd
    import numpy as np
    
    print("\n1. Creating sample market data...")
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='h')
    
    for symbol in agent.symbols:
        # Generate realistic OHLC data
        np.random.seed(42)  # For reproducible results
        price_base = 1.1000 if symbol == 'EURUSD' else 1.2500
        
        # Random walk with trend
        returns = np.random.normal(0, 0.0001, len(dates))
        trend = np.linspace(0, 0.05, len(dates))  # 5% trend over the year
        price_changes = returns + trend / len(dates)
        
        prices = price_base + np.cumsum(price_changes)
        
        # Create OHLC data
        data = pd.DataFrame({
            'open': prices,
            'high': prices + np.random.uniform(0, 0.001, len(prices)),
            'low': prices - np.random.uniform(0, 0.001, len(prices)),
            'close': prices,
            'volume': np.random.randint(100, 1000, len(prices))  # CHANGED: 'tick_volume' → 'volume'
        }, index=dates)
        
        # Store data
        key = f"{symbol}_60"  # H1 timeframe
        agent.feature_data[key] = data
    
    print("✓ Sample data created")
    
    print("\n2. Initializing RL agents...")
    agent.initialize_agents()
    print("✓ Agents initialized")
    
    print("\n3. Running training simulation...")
    training_results = agent.train_agents(episodes=20)  # Shorter for demo
    
    print("\n4. Running backtest simulation...")
    backtest_results = agent.run_backtests()
    
    print("\n5. Performance Summary:")
    for key, result in backtest_results.items():
        if result:
            metrics = result['performance_metrics']
            print(f"\n{key}:")
            print(f"  Total Return: {metrics.get('total_return', 0)*100:.2f}%")
            print(f"  Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
            print(f"  Total Trades: {metrics.get('total_trades', 0)}")
            print(f"  Max Drawdown: {result['risk_metrics'].get('max_drawdown', 0)*100:.2f}%")
    
    print("\n✓ Demo completed successfully!")
    print("\nTo run with real MT5 data, use: python main.py --mode train")

def run_training_mode(agent: MT5RLAgent, episodes: int):
    """Run in training mode"""
    print(f"=== Training Mode - {episodes} Episodes ===")
    
    # Connect to MT5
    if not agent.connect_mt5():
        print("Failed to connect to MT5. Check your connection settings.")
        return
    
    print("Connected to MT5")
    
    # Prepare training data
    print("Preparing training data...")
    for symbol in agent.symbols:
        for timeframe in agent.timeframes:
            try:
                agent.prepare_training_data(symbol, timeframe, 5000)
                print(f"✓ Data prepared for {symbol} on timeframe {timeframe}")
            except Exception as e:
                print(f"✗ Failed to prepare data for {symbol}-{timeframe}: {e}")
    
    # Initialize agents
    print("Initializing agents...")
    agent.initialize_agents()
    
    # Start training
    print(f"Starting training for {episodes} episodes...")
    training_results = agent.train_agents(episodes)
    
    print("Training completed!")
    
    # Save configuration
    config_path = f"config/agent_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("config", exist_ok=True)
    agent.save_configuration(config_path)
    print(f"Configuration saved to {config_path}")

def run_backtest_mode(agent: MT5RLAgent):
    """Run comprehensive backtesting"""
    print("=== Backtest Mode ===")
    
    # Connect to MT5
    if not agent.connect_mt5():
        print("Failed to connect to MT5. Check your connection settings.")
        return
    
    # Prepare data
    print("Preparing data for backtesting...")
    for symbol in agent.symbols:
        for timeframe in agent.timeframes:
            try:
                agent.prepare_training_data(symbol, timeframe, 10000)  # More data for backtesting
            except Exception as e:
                print(f"Failed to prepare data for {symbol}-{timeframe}: {e}")
    
    # Initialize agents
    agent.initialize_agents()
    
    # Load trained models if available
    model_dir = "models"
    if os.path.exists(model_dir):
        for key in agent.agents.keys():
            model_path = os.path.join(model_dir, f"{key}_final.pkl")
            if os.path.exists(model_path):
                try:
                    agent.agents[key].load_model(model_path)
                    print(f"✓ Loaded trained model for {key}")
                except Exception as e:
                    print(f"✗ Failed to load model for {key}: {e}")
    
    # Run backtests
    print("Running backtests...")
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 1 year back
    backtest_results = agent.run_backtests(start_date=start_date)
    
    # Print summary
    print("\n=== Backtest Results Summary ===")
    for key, result in backtest_results.items():
        if result:
            metrics = result['performance_metrics']
            risk_metrics = result['risk_metrics']
            print(f"\n{key}:")
            print(f"  Period: {result['start_date']} to {result['end_date']}")
            print(f"  Total Return: {metrics.get('total_return', 0)*100:.2f}%")
            print(f"  Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 0):.3f}")
            print(f"  Max Drawdown: {risk_metrics.get('max_drawdown', 0)*100:.2f}%")
            print(f"  Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")  # Use .get() for safety
            print(f"  Total Trades: {metrics.get('total_trades', 0)}")

def run_live_mode(agent: MT5RLAgent, auto_trade: bool):
    """Run in live trading mode"""
    print("=== Live Trading Mode ===")
    print(f"Auto-trade: {'Enabled' if auto_trade else 'Disabled'}")
    
    # Connect to MT5
    if not agent.connect_mt5():
        print("Failed to connect to MT5. Check your connection settings.")
        return
    
    # Load trained models
    model_dir = "models"
    if not os.path.exists(model_dir):
        print("No trained models found. Please run training first.")
        return
    
    # Prepare initial data
    for symbol in agent.symbols:
        for timeframe in agent.timeframes:
            try:
                agent.prepare_training_data(symbol, timeframe, 1000)
            except Exception as e:
                print(f"Failed to prepare data for {symbol}-{timeframe}: {e}")
    
    # Initialize agents
    agent.initialize_agents()
    
    # Load models
    models_loaded = 0
    for key in agent.agents.keys():
        model_path = os.path.join(model_dir, f"{key}_final.pkl")
        if os.path.exists(model_path):
            try:
                agent.agents[key].load_model(model_path)
                models_loaded += 1
                print(f"✓ Loaded model for {key}")
            except Exception as e:
                print(f"✗ Failed to load model for {key}: {e}")
    
    if models_loaded == 0:
        print("No models could be loaded. Exiting.")
        return
    
    print(f"Loaded {models_loaded} trained models")
    
    # Start live trading
    agent.start_live_trading(auto_trade=auto_trade)
    
    print("Live trading started. Press Ctrl+C to stop.")
    print("Monitoring market conditions...")
    
    try:
        while True:
            # Print periodic status
            import time
            time.sleep(30)  # Update every 30 seconds
            
            summary = agent.get_performance_summary()
            print(f"\nStatus Update - {summary['timestamp'].strftime('%H:%M:%S')}")
            print(f"Open Positions: {summary['account_status']['open_positions']}")
            print(f"Unrealized P&L: ${summary['account_status']['unrealized_pnl']:.2f}")
            print(f"Total Trades: {summary['trade_analysis']['total_trades']}")
            
            # Print risk alerts
            risk_alerts = summary['risk_summary']['risk_alerts']
            if risk_alerts:
                print("⚠️  Risk Alerts:")
                for alert in risk_alerts[-3:]:  # Show last 3 alerts
                    print(f"  - {alert}")
            
    except KeyboardInterrupt:
        print("\nStopping live trading...")
        agent.stop_live_trading()

if __name__ == "__main__":
    main()