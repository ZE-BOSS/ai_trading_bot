"""
MetaTrader 5 Connector for Real-time Data and Trade Execution
"""

import json
import random
import math
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import MetaTrader5 as mt5
import pandas as pd

class MT5Connector:
    def __init__(self, login: int = None, password: str = None, server: str = None, demo_mode: bool = False):
        """Initialize MT5 connection"""
        self.login = login
        self.password = password
        self.server = server
        self.connected = False
        self.demo_mode = demo_mode
        self.lock = threading.Lock()
        self.monitoring_thread = None
        self.is_monitoring = False
        self.live_data = {}
        self.trade_history = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if not demo_mode:
            self._connect()
        
    def _connect(self):
        """Connect to MetaTrader 5"""
        try:
            import MetaTrader5 as mt5
            if not mt5.initialize():
                raise Exception(f"MT5 initialization failed: {mt5.last_error()}")
            self.connected = True
            self.logger.info("Successfully connected to MT5")
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            self.connected = False
    
    def connect(self) -> bool:
        """Establish connection to MT5"""
        try:
            if not mt5.initialize():
                self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
                
            if self.login and self.password and self.server:
                if not mt5.login(self.login, self.password, self.server):
                    self.logger.error(f"MT5 login failed: {mt5.last_error()}")
                    return False
                    
            self.connected = True
            self.logger.info("Successfully connected to MT5")
            return True
            
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MT5"""
        self.stop_monitoring()
        mt5.shutdown()
        self.connected = False
        self.logger.info("Disconnected from MT5")
    
    def get_market_data(self, symbol: str, timeframe: str, count: int = 100) -> Dict:
        """Get market data for specified symbol and timeframe"""
        if self.demo_mode:
            # Generate synthetic market data using only standard library
            current_time = datetime.now()
            dates = [current_time - timedelta(minutes=i) for i in range(count-1, -1, -1)]
            
            # Simple random walk for price generation
            base_price = 1.1000
            prices = []
            current_price = base_price
            
            for _ in range(count):
                change = random.gauss(0, 0.0001)  # Random walk with small steps
                current_price += change
                prices.append(current_price)
            
            return {
                'time': dates,
                'open': prices,
                'high': [p + random.random() * 0.0005 for p in prices],
                'low': [p - random.random() * 0.0005 for p in prices],
                'close': prices,
                'volume': [random.randint(100, 1000) for _ in range(count)]
            }
        
        else:
            try:
                import MetaTrader5 as mt5
                import pandas as pd
                # Convert timeframe string to MT5 constant
                tf_map = {
                    'M1': mt5.TIMEFRAME_M1,
                    'M5': mt5.TIMEFRAME_M5,
                    'M15': mt5.TIMEFRAME_M15,
                    'M30': mt5.TIMEFRAME_M30,
                    'H1': mt5.TIMEFRAME_H1,
                    'H4': mt5.TIMEFRAME_H4,
                    'D1': mt5.TIMEFRAME_D1
                }
                
                timeframe_mt5 = tf_map.get(timeframe, mt5.TIMEFRAME_M1)
                rates = mt5.copy_rates_from_pos(symbol, timeframe_mt5, 0, count)
                
                if rates is None:
                    raise Exception(f"Failed to get rates: {mt5.last_error()}")
                
                # Convert to simple dict without pandas
                data = {
                    'time': [datetime.fromtimestamp(r['time']) for r in rates],
                    'open': [r['open'] for r in rates],
                    'high': [r['high'] for r in rates],
                    'low': [r['low'] for r in rates],
                    'close': [r['close'] for r in rates],
                    'volume': [r['tick_volume'] for r in rates]
                }
                
                return data
                
            except Exception as e:
                print(f"Error getting market data: {e}")
                return {}
    
    def get_historical_data(self, symbol: str, timeframe: int, count: int = 1000) -> pd.DataFrame:
        """Retrieve historical market data"""
        if self.demo_mode:
            # Generate demo data as DataFrame
            current_time = datetime.now()
            dates = [current_time - timedelta(minutes=i) for i in range(count-1, -1, -1)]
            base_price = 1.1000
            prices = [base_price + random.gauss(0, 0.0001) for _ in range(count)]
            data = {
                'time': dates,
                'open': prices,
                'high': [p + random.random() * 0.0005 for p in prices],
                'low': [p - random.random() * 0.0005 for p in prices],
                'close': prices,
                'volume': [random.randint(100, 1000) for _ in range(count)]
            }
            
            return pd.DataFrame(data).set_index('time')
        
        if not self.connected:
            raise ConnectionError("Not connected to MT5")
            
        try:
            import MetaTrader5 as mt5
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

            if rates is None:
                self.logger.error(f"Failed to get data for {symbol}")
                return pd.DataFrame()  # Return empty DataFrame
                
            # Convert to DataFrame
            data = pd.DataFrame(rates)
            data['time'] = pd.to_datetime(data['time'], unit='s')
            data.set_index('time', inplace=True)
            
            return data
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()  # Return empty DataFrame
    
    def get_live_tick(self, symbol: str) -> Dict:
        """Get current tick data"""
        if not self.connected:
            raise ConnectionError("Not connected to MT5")
            
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {}
            
        return {
            'symbol': symbol,
            'bid': tick.bid,
            'ask': tick.ask,
            'spread': tick.ask - tick.bid,
            'volume': tick.volume,
            'time': datetime.fromtimestamp(tick.time)
        }
    
    def start_monitoring(self, symbols: List[str], callback=None):
        """Start real-time monitoring thread"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(symbols, callback)
        )
        self.monitoring_thread.start()
        self.logger.info(f"Started monitoring {symbols}")
    
    def stop_monitoring(self):
        """Stop monitoring thread"""
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join()
            
    def _monitor_loop(self, symbols: List[str], callback=None):
        """Internal monitoring loop"""
        while self.is_monitoring:
            try:
                for symbol in symbols:
                    tick_data = self.get_live_tick(symbol)
                    if tick_data:
                        self.live_data[symbol] = tick_data
                        if callback:
                            callback(tick_data)
                            
                time.sleep(0.1)  # 100ms update rate
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(1)
    
    def place_order(self, symbol: str, order_type: str, volume: float, 
                   price: float = None, sl: float = None, tp: float = None, comment: str = "") -> Dict:
        """Place a trading order"""
        if self.demo_mode:
            return {
                'order': random.randint(100000, 999999),
                'result': 'success',
                'message': 'Demo order placed successfully'
            }
        
        try:
            import MetaTrader5 as mt5
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            if price is not None:
                request["price"] = price
            if sl is not None:
                request["sl"] = sl
            if tp is not None:
                request["tp"] = tp
                
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    "success": False, 
                    "error": f"Order failed: {result.retcode}",
                    "result": result
                }
                
            return {
                "success": True,
                "ticket": result.order,
                "volume": result.volume,
                "price": result.price,
                "result": result
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_positions(self) -> List[Dict]:
        """Get current open positions"""
        if not self.connected:
            return []
            
        positions = mt5.positions_get()
        if positions is None:
            return []
            
        return [
            {
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "type": pos.type,
                "volume": pos.volume,
                "price_open": pos.price_open,
                "price_current": pos.price_current,
                "profit": pos.profit,
                "sl": pos.sl,
                "tp": pos.tp,
                "time": datetime.fromtimestamp(pos.time)
            }
            for pos in positions
        ]
    
    def close_position(self, ticket: int) -> Dict:
        """Close a specific position"""
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return {"success": False, "error": "Position not found"}
            
        position = positions[0]
        
        # Determine opposite order type
        order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": order_type,
            "position": ticket,
            "comment": "Close by agent",
        }
        
        result = mt5.order_send(close_request)
        
        return {
            "success": result.retcode == mt5.TRADE_RETCODE_DONE,
            "result": result
        }