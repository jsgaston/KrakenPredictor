import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import requests
import json
import numpy as np
from datetime import datetime
import threading
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches

class TechnicalIndicators:
    """Indicadores técnicos para análisis de mercado"""
    
    @staticmethod
    def sma(data, period):
        """Simple Moving Average"""
        if len(data) < period:
            return [None] * len(data)
        sma_values = []
        for i in range(len(data)):
            if i < period - 1:
                sma_values.append(None)
            else:
                sma_values.append(sum(data[i-period+1:i+1]) / period)
        return sma_values
    
    @staticmethod
    def ema(data, period):
        """Exponential Moving Average"""
        if len(data) < period:
            return [None] * len(data)
        
        ema_values = [None] * (period - 1)
        multiplier = 2 / (period + 1)
        ema_values.append(sum(data[:period]) / period)
        
        for i in range(period, len(data)):
            ema = (data[i] - ema_values[-1]) * multiplier + ema_values[-1]
            ema_values.append(ema)
        
        return ema_values
    
    @staticmethod
    def rsi(data, period=14):
        """Relative Strength Index"""
        if len(data) < period + 1:
            return [None] * len(data)
        
        deltas = [data[i] - data[i-1] for i in range(1, len(data))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        # Use EMA for smoothing instead of SMA
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        rsi_values = [None] * (period + 1)
        
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi_values.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(100 - (100 / (1 + rs)))
        
        return rsi_values
    
    @staticmethod
    def bollinger_bands(data, period=20, std_dev=2):
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, period)
        
        upper_band = []
        lower_band = []
        
        for i in range(len(data)):
            if i < period - 1 or sma[i] is None:
                upper_band.append(None)
                lower_band.append(None)
            else:
                std = np.std(data[i-period+1:i+1])
                upper_band.append(sma[i] + (std_dev * std))
                lower_band.append(sma[i] - (std_dev * std))
        
        return sma, upper_band, lower_band
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        
        macd_line = []
        for i in range(len(data)):
            if ema_fast[i] is None or ema_slow[i] is None:
                macd_line.append(None)
            else:
                macd_line.append(ema_fast[i] - ema_slow[i])
        
        # Calculate signal line properly
        first_valid = next((i for i, v in enumerate(macd_line) if v is not None), None)
        if first_valid is None:
            return macd_line, [None] * len(macd_line), [None] * len(macd_line)
        
        valid_macd = [v for v in macd_line if v is not None]
        signal_ema = TechnicalIndicators.ema(valid_macd, signal)
        
        signal_line = [None] * first_valid + signal_ema
        
        histogram = []
        for i in range(len(macd_line)):
            if macd_line[i] is None or signal_line[i] is None:
                histogram.append(None)
            else:
                histogram.append(macd_line[i] - signal_line[i])
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def ichimoku(data, tenkan=9, kijun=26, senkou_b=52, displacement=26):
        """Ichimoku Cloud"""
        def calculate_line(period):
            line = []
            for i in range(len(data)):
                if i < period - 1:
                    line.append(None)
                else:
                    period_data = data[i-period+1:i+1]
                    line.append((max(period_data) + min(period_data)) / 2)
            return line
        
        # Tenkan-sen (Conversion Line)
        tenkan_sen = calculate_line(tenkan)
        
        # Kijun-sen (Base Line)
        kijun_sen = calculate_line(kijun)
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = []
        for i in range(len(data)):
            if tenkan_sen[i] is None or kijun_sen[i] is None:
                senkou_span_a.append(None)
            else:
                senkou_span_a.append((tenkan_sen[i] + kijun_sen[i]) / 2)
        
        # Senkou Span B (Leading Span B)
        senkou_span_b = calculate_line(senkou_b)
        
        # Chikou Span (Lagging Span) - price displaced backwards
        chikou_span = [None] * displacement + data[:-displacement] if len(data) > displacement else [None] * len(data)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    @staticmethod
    def stochastic(high, low, close, k_period=14, d_period=3):
        """Stochastic Oscillator"""
        k_values = []
        
        for i in range(len(close)):
            if i < k_period - 1:
                k_values.append(None)
            else:
                period_high = max(high[i-k_period+1:i+1])
                period_low = min(low[i-k_period+1:i+1])
                
                if period_high == period_low:
                    k_values.append(50)
                else:
                    k = ((close[i] - period_low) / (period_high - period_low)) * 100
                    k_values.append(k)
        
        # %D is SMA of %K
        d_values = TechnicalIndicators.sma(k_values, d_period)
        
        return k_values, d_values
    
    @staticmethod
    def atr(high, low, close, period=14):
        """Average True Range"""
        tr_values = []
        
        for i in range(len(close)):
            if i == 0:
                tr = high[i] - low[i]
            else:
                tr = max(
                    high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1])
                )
            tr_values.append(tr)
        
        return TechnicalIndicators.sma(tr_values, period)
    
    @staticmethod
    def cci(high, low, close, period=20):
        """Commodity Channel Index"""
        typical_price = [(h + l + c) / 3 for h, l, c in zip(high, low, close)]
        sma_tp = TechnicalIndicators.sma(typical_price, period)
        
        cci_values = []
        for i in range(len(typical_price)):
            if i < period - 1 or sma_tp[i] is None:
                cci_values.append(None)
            else:
                period_tp = typical_price[i-period+1:i+1]
                mean_deviation = sum(abs(tp - sma_tp[i]) for tp in period_tp) / period
                
                if mean_deviation == 0:
                    cci_values.append(0)
                else:
                    cci = (typical_price[i] - sma_tp[i]) / (0.015 * mean_deviation)
                    cci_values.append(cci)
        
        return cci_values
    
    @staticmethod
    def adx(high, low, close, period=14):
        """Average Directional Index"""
        if len(close) < period + 1:
            return [None] * len(close), [None] * len(close), [None] * len(close)
        
        tr_values = []
        plus_dm = []
        minus_dm = []
        
        for i in range(len(close)):
            if i == 0:
                tr_values.append(high[i] - low[i])
                plus_dm.append(0)
                minus_dm.append(0)
            else:
                # True Range
                tr = max(
                    high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1])
                )
                tr_values.append(tr)
                
                # Directional Movement
                up_move = high[i] - high[i-1]
                down_move = low[i-1] - low[i]
                
                if up_move > down_move and up_move > 0:
                    plus_dm.append(up_move)
                else:
                    plus_dm.append(0)
                
                if down_move > up_move and down_move > 0:
                    minus_dm.append(down_move)
                else:
                    minus_dm.append(0)
        
        # Smooth TR, +DM, -DM using Wilder's smoothing
        smoothed_tr = [None] * (period - 1)
        smoothed_plus = [None] * (period - 1)
        smoothed_minus = [None] * (period - 1)
        
        smoothed_tr.append(sum(tr_values[:period]))
        smoothed_plus.append(sum(plus_dm[:period]))
        smoothed_minus.append(sum(minus_dm[:period]))
        
        for i in range(period, len(close)):
            smoothed_tr.append(smoothed_tr[-1] - (smoothed_tr[-1] / period) + tr_values[i])
            smoothed_plus.append(smoothed_plus[-1] - (smoothed_plus[-1] / period) + plus_dm[i])
            smoothed_minus.append(smoothed_minus[-1] - (smoothed_minus[-1] / period) + minus_dm[i])
        
        # Calculate +DI and -DI
        plus_di = []
        minus_di = []
        
        for i in range(len(smoothed_tr)):
            if smoothed_tr[i] is None or smoothed_tr[i] == 0:
                plus_di.append(None)
                minus_di.append(None)
            else:
                plus_di.append(100 * smoothed_plus[i] / smoothed_tr[i])
                minus_di.append(100 * smoothed_minus[i] / smoothed_tr[i])
        
        # Calculate DX
        dx_values = []
        for i in range(len(plus_di)):
            if plus_di[i] is None or minus_di[i] is None:
                dx_values.append(None)
            else:
                di_sum = plus_di[i] + minus_di[i]
                if di_sum == 0:
                    dx_values.append(0)
                else:
                    dx = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
                    dx_values.append(dx)
        
        # Calculate ADX using Wilder's smoothing
        adx_values = [None] * (period * 2 - 2)
        
        first_valid = next((i for i, v in enumerate(dx_values) if v is not None), None)
        if first_valid is not None and first_valid + period <= len(dx_values):
            valid_dx = [v for v in dx_values[first_valid:] if v is not None]
            if len(valid_dx) >= period:
                adx_values.extend([None] * (first_valid))
                adx_values.append(sum(valid_dx[:period]) / period)
                
                for i in range(period, len(valid_dx)):
                    new_adx = (adx_values[-1] * (period - 1) + valid_dx[i]) / period
                    adx_values.append(new_adx)
        
        # Pad to correct length
        while len(adx_values) < len(close):
            adx_values.append(None)
        
        return adx_values[:len(close)], plus_di, minus_di
    
    @staticmethod
    def obv(close, volume):
        """On-Balance Volume"""
        obv_values = [0]
        
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv_values.append(obv_values[-1] + volume[i])
            elif close[i] < close[i-1]:
                obv_values.append(obv_values[-1] - volume[i])
            else:
                obv_values.append(obv_values[-1])
        
        return obv_values
    
    @staticmethod
    def williams_r(high, low, close, period=14):
        """Williams %R"""
        wr_values = []
        
        for i in range(len(close)):
            if i < period - 1:
                wr_values.append(None)
            else:
                period_high = max(high[i-period+1:i+1])
                period_low = min(low[i-period+1:i+1])
                
                if period_high == period_low:
                    wr_values.append(-50)
                else:
                    wr = ((period_high - close[i]) / (period_high - period_low)) * -100
                    wr_values.append(wr)
        
        return wr_values

class HopfieldNetwork:
    """Hopfield Network - Nobel Prize in Physics 2024 (John Hopfield)"""
    
    def __init__(self, pattern_size):
        self.pattern_size = pattern_size
        self.weights = np.zeros((pattern_size, pattern_size))
        self.patterns_learned = []
        
    def pattern_to_binary(self, pattern):
        """Converts HH/HL/LH/LL pattern to bipolar binary representation (-1, +1)"""
        binary = []
        for char in pattern:
            binary.append(1 if char == 'H' else -1)
        return np.array(binary)
    
    def binary_to_pattern(self, binary):
        """Converts bipolar binary back to HH/HL/LH/LL pattern"""
        pattern = ''
        for val in binary:
            pattern += 'H' if val > 0 else 'L'
        return pattern
    
    def train(self, patterns):
        """Trains the Hopfield network with historical patterns"""
        self.patterns_learned = patterns
        self.weights = np.zeros((self.pattern_size, self.pattern_size))
        
        binary_patterns = [self.pattern_to_binary(p) for p in patterns]
        
        for pattern in binary_patterns:
            pattern = pattern.reshape(-1, 1)
            self.weights += np.dot(pattern, pattern.T)
        
        self.weights /= len(patterns)
        np.fill_diagonal(self.weights, 0)
        
        return len(patterns)
    
    def energy(self, state):
        """Calculates system energy (Lyapunov function)"""
        return -0.5 * np.dot(state, np.dot(self.weights, state))
    
    def recall(self, partial_pattern, max_iterations=100):
        """Retrieves complete pattern from partial information"""
        state = self.pattern_to_binary(partial_pattern)
        
        for iteration in range(max_iterations):
            old_state = state.copy()
            
            for i in range(self.pattern_size):
                activation = np.dot(self.weights[i], state)
                state[i] = 1 if activation >= 0 else -1
            
            if np.array_equal(state, old_state):
                break
        
        return self.binary_to_pattern(state)
    
    def predict_next(self, current_pattern, historical_transitions):
        """Predicts next pattern using Hopfield Network"""
        similar_patterns = []
        for i in range(len(historical_transitions) - 1):
            if historical_transitions[i] == current_pattern:
                similar_patterns.append(historical_transitions[i + 1])
        
        if not similar_patterns:
            return self.recall(current_pattern)
        
        pattern_counts = Counter(similar_patterns)
        most_common = pattern_counts.most_common(1)[0][0]
        
        refined_prediction = self.recall(most_common)
        
        confidence = (pattern_counts[most_common] / len(similar_patterns)) * 100
        
        return refined_prediction, confidence, len(similar_patterns)

class ModernButton(tk.Canvas):
    """Custom modern rounded button with gradient effect"""
    def __init__(self, parent, text, command, bg_color, fg_color, width=140, height=40):
        super().__init__(parent, width=width, height=height, bg=parent['bg'], 
                        highlightthickness=0, cursor='hand2')
        self.command = command
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.text = text
        self.width = width
        self.height = height
        
        self.draw_button()
        self.bind('<Button-1>', lambda e: self.on_click())
        self.bind('<Enter>', lambda e: self.on_hover())
        self.bind('<Leave>', lambda e: self.on_leave())
        
    def draw_button(self, hover=False):
        self.delete('all')
        color = self.lighten_color(self.bg_color) if hover else self.bg_color
        
        r = 10
        self.create_arc(0, 0, 2*r, 2*r, start=90, extent=90, fill=color, outline='')
        self.create_arc(self.width-2*r, 0, self.width, 2*r, start=0, extent=90, fill=color, outline='')
        self.create_arc(0, self.height-2*r, 2*r, self.height, start=180, extent=90, fill=color, outline='')
        self.create_arc(self.width-2*r, self.height-2*r, self.width, self.height, start=270, extent=90, fill=color, outline='')
        
        self.create_rectangle(r, 0, self.width-r, self.height, fill=color, outline='')
        self.create_rectangle(0, r, self.width, self.height-r, fill=color, outline='')
        
        self.create_text(self.width/2, self.height/2, text=self.text, 
                        fill=self.fg_color, font=('Segoe UI', 10, 'bold'))
    
    def lighten_color(self, color):
        """Lighten color for hover effect"""
        colors = {
            '#3b82f6': '#60a5fa',
            '#10b981': '#34d399',
            '#8b5cf6': '#a78bfa',
            '#f59e0b': '#fbbf24',
            '#ef4444': '#f87171'
        }
        return colors.get(color, color)
    
    def on_hover(self):
        self.draw_button(hover=True)
    
    def on_leave(self):
        self.draw_button(hover=False)
    
    def on_click(self):
        if self.command:
            self.command()

class KrakenPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Kraken Predictor - Hopfield Network + Technical Indicators")
        self.root.geometry("1600x900")
        
        # Modern dark theme
        self.root.configure(bg='#0a0e27')
        
        # Auto-refresh variables
        self.auto_refresh = tk.BooleanVar(value=False)
        self.refresh_interval = tk.StringVar(value='60')
        self.refresh_job = None
        
        # Variables
        self.selected_pair = tk.StringVar(value='XXBTZUSD')
        self.interval = tk.StringVar(value='60')
        self.market_type = tk.StringVar(value='spot')
        self.search_var = tk.StringVar()
        self.pattern_count = tk.StringVar(value='100')
        self.chart_size = tk.StringVar(value='Medium')
        
        # Indicator variables
        self.indicators = {
            'SMA_20': tk.BooleanVar(value=False),
            'SMA_50': tk.BooleanVar(value=False),
            'SMA_200': tk.BooleanVar(value=False),
            'EMA_12': tk.BooleanVar(value=False),
            'EMA_26': tk.BooleanVar(value=False),
            'EMA_50': tk.BooleanVar(value=False),
            'BB': tk.BooleanVar(value=False),
            'RSI': tk.BooleanVar(value=False),
            'MACD': tk.BooleanVar(value=False),
            'Stochastic': tk.BooleanVar(value=False),
            'Volume': tk.BooleanVar(value=False),
            'Ichimoku': tk.BooleanVar(value=False),
            'ATR': tk.BooleanVar(value=False),
            'CCI': tk.BooleanVar(value=False),
            'ADX': tk.BooleanVar(value=False),
            'OBV': tk.BooleanVar(value=False),
            'WilliamsR': tk.BooleanVar(value=False)
        }
        
        # Data
        self.all_pairs = {}
        self.filtered_pairs = []
        self.market_data = []
        self.patterns = []
        self.prediction = None
        
        # Hopfield Network
        self.hopfield_net = HopfieldNetwork(pattern_size=2)
        self.hopfield_trained = False
        
        # Create UI
        self.create_ui()
        
        # Auto load pairs
        self.load_trading_pairs()
    
    def create_rounded_frame(self, parent, bg='#1a1f3a', padx=15, pady=15):
        """Create a modern rounded frame"""
        frame = tk.Frame(parent, bg=bg, relief=tk.FLAT, bd=0)
        return frame
    
    def create_ui(self):
        # Main container with padding
        main_container = tk.Frame(self.root, bg='#0a0e27')
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # ========== COMPACT HEADER ==========
        header_frame = self.create_rounded_frame(main_container, bg='#1e3a8a')
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        header_inner = tk.Frame(header_frame, bg='#1e3a8a')
        header_inner.pack(padx=15, pady=12)
        
        tk.Label(header_inner, text="🏆 HOPFIELD NETWORK + TECHNICAL ANALYSIS", 
                bg='#1e3a8a', fg='#ffffff', 
                font=('Segoe UI', 14, 'bold')).pack()
        tk.Label(header_inner, text="Nobel Prize 2024 • Advanced Market Prediction", 
                bg='#1e3a8a', fg='#93c5fd', 
                font=('Segoe UI', 9)).pack()
        
        # ========== TOP SECTION: CONFIG + INDICATORS ==========
        top_section = tk.Frame(main_container, bg='#0a0e27')
        top_section.pack(fill=tk.X, pady=(0, 10))
        
        # Left: Configuration
        config_frame = self.create_rounded_frame(top_section, bg='#1a1f3a')
        config_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        
        config_inner = tk.Frame(config_frame, bg='#1a1f3a')
        config_inner.pack(padx=15, pady=15, fill=tk.BOTH, expand=True)
        
        tk.Label(config_inner, text="⚙️ Configuration", bg='#1a1f3a', fg='#ffffff', 
                font=('Segoe UI', 12, 'bold')).grid(row=0, column=0, columnspan=4, 
                                                     sticky='w', pady=(0, 10))
        
        # Market Type
        tk.Label(config_inner, text="Market Type", bg='#1a1f3a', fg='#94a3b8', 
                font=('Segoe UI', 9)).grid(row=1, column=0, sticky='w', pady=6)
        
        market_frame = tk.Frame(config_inner, bg='#1a1f3a')
        market_frame.grid(row=1, column=1, sticky='w', pady=6, padx=8)
        
        for market in ['spot', 'margin', 'futures']:
            rb = tk.Radiobutton(market_frame, text=market.upper(), variable=self.market_type, 
                              value=market, bg='#1a1f3a', fg='#ffffff', 
                              selectcolor='#3b82f6', font=('Segoe UI', 8),
                              activebackground='#1a1f3a', activeforeground='#60a5fa',
                              command=self.load_trading_pairs)
            rb.pack(side=tk.LEFT, padx=4)
        
        # Search Pair
        tk.Label(config_inner, text="Search Pair", bg='#1a1f3a', fg='#94a3b8', 
                font=('Segoe UI', 9)).grid(row=2, column=0, sticky='w', pady=6)
        
        search_entry = tk.Entry(config_inner, textvariable=self.search_var, 
                               width=20, bg='#0f1629', fg='#ffffff', 
                               insertbackground='#3b82f6', font=('Segoe UI', 9),
                               relief=tk.FLAT, bd=0)
        search_entry.grid(row=2, column=1, padx=8, pady=6, sticky='ew', ipady=6)
        search_entry.bind('<KeyRelease>', self.filter_pairs)
        
        # Trading Pair
        tk.Label(config_inner, text="Trading Pair", bg='#1a1f3a', fg='#94a3b8', 
                font=('Segoe UI', 9)).grid(row=3, column=0, sticky='w', pady=6)
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Modern.TCombobox', 
                       fieldbackground='#0f1629',
                       background='#0f1629',
                       foreground='#ffffff',
                       borderwidth=0)
        
        self.pair_combo = ttk.Combobox(config_inner, textvariable=self.selected_pair, 
                                       state='readonly', width=18, 
                                       style='Modern.TCombobox',
                                       font=('Segoe UI', 9))
        self.pair_combo.grid(row=3, column=1, padx=8, pady=6, sticky='w')
        
        # Interval
        tk.Label(config_inner, text="Timeframe", bg='#1a1f3a', fg='#94a3b8', 
                font=('Segoe UI', 9)).grid(row=4, column=0, sticky='w', pady=6)
        
        interval_frame = tk.Frame(config_inner, bg='#1a1f3a')
        interval_frame.grid(row=4, column=1, columnspan=3, sticky='w', pady=6, padx=8)
        
        intervals = [
            ('1m', '1'), ('5m', '5'), ('15m', '15'), ('1h', '60'), ('4h', '240'), ('1d', '1440')
        ]
        
        for text, value in intervals:
            rb = tk.Radiobutton(interval_frame, text=text, variable=self.interval, 
                              value=value, bg='#1a1f3a', fg='#ffffff', 
                              selectcolor='#3b82f6', font=('Segoe UI', 8),
                              activebackground='#1a1f3a', activeforeground='#60a5fa')
            rb.pack(side=tk.LEFT, padx=2)
        
        # Pattern Count
        tk.Label(config_inner, text="Training Patterns", bg='#1a1f3a', fg='#94a3b8', 
                font=('Segoe UI', 9)).grid(row=5, column=0, sticky='w', pady=6)
        
        pattern_entry = tk.Entry(config_inner, textvariable=self.pattern_count, 
                                width=8, bg='#0f1629', fg='#ffffff', 
                                insertbackground='#3b82f6', font=('Segoe UI', 9),
                                relief=tk.FLAT, bd=0)
        pattern_entry.grid(row=5, column=1, padx=8, pady=6, sticky='w', ipady=6)
        
        # Network Status
        self.network_status = tk.Label(config_inner, text="⚪ Network: Not Trained", 
                                       bg='#1a1f3a', fg='#ef4444', 
                                       font=('Segoe UI', 9, 'bold'))
        self.network_status.grid(row=6, column=0, columnspan=2, pady=(10, 8), sticky='w')
        
        # Action Buttons
        btn_frame = tk.Frame(config_inner, bg='#1a1f3a')
        btn_frame.grid(row=7, column=0, columnspan=4, pady=(8, 0), sticky='w')
        
        ModernButton(btn_frame, "📊 Analyze", self.analyze_market, 
                    '#3b82f6', '#ffffff', 120, 38).pack(side=tk.LEFT, padx=3)
        ModernButton(btn_frame, "🔄 Update", self.update_chart_only, 
                    '#10b981', '#ffffff', 120, 38).pack(side=tk.LEFT, padx=3)
        ModernButton(btn_frame, "💾 Export", self.export_data, 
                    '#8b5cf6', '#ffffff', 120, 38).pack(side=tk.LEFT, padx=3)
        
        # Auto-refresh controls
        auto_frame = tk.Frame(config_inner, bg='#1a1f3a')
        auto_frame.grid(row=8, column=0, columnspan=4, pady=(8, 0), sticky='w')
        
        self.auto_refresh_cb = tk.Checkbutton(auto_frame, text="Auto-Refresh", 
                                              variable=self.auto_refresh,
                                              bg='#1a1f3a', fg='#22c55e', 
                                              selectcolor='#1a1f3a',
                                              font=('Segoe UI', 9, 'bold'),
                                              activebackground='#1a1f3a',
                                              activeforeground='#22c55e',
                                              command=self.toggle_auto_refresh)
        self.auto_refresh_cb.pack(side=tk.LEFT, padx=5)
        
        tk.Label(auto_frame, text="Every:", bg='#1a1f3a', fg='#94a3b8', 
                font=('Segoe UI', 8)).pack(side=tk.LEFT, padx=(10, 3))
        
        refresh_combo = ttk.Combobox(auto_frame, textvariable=self.refresh_interval,
                                    values=['30', '60', '120', '300', '600'],
                                    state='readonly', width=5,
                                    style='Modern.TCombobox',
                                    font=('Segoe UI', 8))
        refresh_combo.pack(side=tk.LEFT, padx=3)
        
        tk.Label(auto_frame, text="sec", bg='#1a1f3a', fg='#94a3b8', 
                font=('Segoe UI', 8)).pack(side=tk.LEFT, padx=3)
        
        # Right: Indicators Panel
        indicators_frame = self.create_rounded_frame(top_section, bg='#1a1f3a')
        indicators_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        ind_inner = tk.Frame(indicators_frame, bg='#1a1f3a')
        ind_inner.pack(padx=15, pady=15, fill=tk.BOTH, expand=True)
        
        tk.Label(ind_inner, text="📈 Technical Indicators", bg='#1a1f3a', fg='#ffffff', 
                font=('Segoe UI', 12, 'bold')).pack(anchor='w', pady=(0, 10))
        
        # Indicators grid
        ind_grid = tk.Frame(ind_inner, bg='#1a1f3a')
        ind_grid.pack(fill=tk.BOTH, expand=True)
        
        indicators_info = [
            ('SMA_20', 'SMA (20)', '#fbbf24'),
            ('SMA_50', 'SMA (50)', '#f59e0b'),
            ('SMA_200', 'SMA (200)', '#ea580c'),
            ('EMA_12', 'EMA (12)', '#10b981'),
            ('EMA_26', 'EMA (26)', '#059669'),
            ('EMA_50', 'EMA (50)', '#047857'),
            ('BB', 'Bollinger', '#8b5cf6'),
            ('Ichimoku', 'Ichimoku', '#06b6d4'),
            ('RSI', 'RSI', '#ef4444'),
            ('MACD', 'MACD', '#3b82f6'),
            ('Stochastic', 'Stochastic', '#ec4899'),
            ('Volume', 'Volume', '#6366f1'),
            ('ATR', 'ATR', '#14b8a6'),
            ('CCI', 'CCI', '#f97316'),
            ('ADX', 'ADX', '#84cc16'),
            ('OBV', 'OBV', '#a855f7'),
            ('WilliamsR', 'Williams %R', '#f43f5e')
        ]
        
        for idx, (key, label, color) in enumerate(indicators_info):
            row = idx // 3
            col = idx % 3
            
            ind_frame = tk.Frame(ind_grid, bg='#0f1629', relief=tk.FLAT, bd=0)
            ind_frame.grid(row=row, column=col, padx=2, pady=2, sticky='ew')
            
            cb = tk.Checkbutton(ind_frame, text=label, variable=self.indicators[key],
                              bg='#0f1629', fg=color, selectcolor='#1a1f3a',
                              font=('Segoe UI', 8, 'bold'), activebackground='#0f1629',
                              activeforeground=color, command=self.update_chart_only)
            cb.pack(anchor='w', padx=8, pady=5)
        
        ind_grid.columnconfigure(0, weight=1)
        ind_grid.columnconfigure(1, weight=1)
        ind_grid.columnconfigure(2, weight=1)
        
        # Quick Actions
        quick_frame = tk.Frame(ind_inner, bg='#1a1f3a')
        quick_frame.pack(fill=tk.X, pady=(10, 0))
        
        ModernButton(quick_frame, "✓ All", self.enable_all_indicators, 
                    '#10b981', '#ffffff', 80, 30).pack(side=tk.LEFT, padx=3)
        ModernButton(quick_frame, "✗ None", self.disable_all_indicators, 
                    '#ef4444', '#ffffff', 80, 30).pack(side=tk.LEFT, padx=3)
        ModernButton(quick_frame, "⭐ Fav", self.set_favorite_indicators, 
                    '#f59e0b', '#ffffff', 80, 30).pack(side=tk.LEFT, padx=3)
        
        # ========== PREDICTION PANEL ==========
        pred_frame = self.create_rounded_frame(main_container, bg='#1a1f3a')
        pred_frame.pack(fill=tk.X, pady=(0, 10))
        
        pred_inner = tk.Frame(pred_frame, bg='#1a1f3a')
        pred_inner.pack(padx=15, pady=12, fill=tk.X)
        
        tk.Label(pred_inner, text="🎯 Hopfield Network Prediction", bg='#1a1f3a', fg='#ffffff', 
                font=('Segoe UI', 11, 'bold')).pack(anchor='w', pady=(0, 8))
        
        cards_frame = tk.Frame(pred_inner, bg='#1a1f3a')
        cards_frame.pack(fill=tk.X)
        
        # Prediction Cards
        self.current_card = self.create_prediction_card(cards_frame, "Current", "--", '#fbbf24')
        self.current_card.pack(side=tk.LEFT, padx=3, fill=tk.BOTH, expand=True)
        
        self.predicted_card = self.create_prediction_card(cards_frame, "Prediction", "--", '#22c55e')
        self.predicted_card.pack(side=tk.LEFT, padx=3, fill=tk.BOTH, expand=True)
        
        self.confidence_card = self.create_prediction_card(cards_frame, "Confidence", "--", '#3b82f6')
        self.confidence_card.pack(side=tk.LEFT, padx=3, fill=tk.BOTH, expand=True)
        
        self.energy_card = self.create_prediction_card(cards_frame, "Energy", "--", '#a78bfa')
        self.energy_card.pack(side=tk.LEFT, padx=3, fill=tk.BOTH, expand=True)
        
        # ========== MAIN CHART ==========
        chart_frame = self.create_rounded_frame(main_container, bg='#1a1f3a')
        chart_frame.pack(fill=tk.BOTH, expand=True)
        
        chart_inner = tk.Frame(chart_frame, bg='#1a1f3a')
        chart_inner.pack(padx=15, pady=15, fill=tk.BOTH, expand=True)
        
        tk.Label(chart_inner, text="📊 Advanced Price Chart", bg='#1a1f3a', fg='#ffffff', 
                font=('Segoe UI', 11, 'bold')).pack(anchor='w', pady=(0, 8))
        
        # Chart container for dynamic sizing
        self.chart_container = tk.Frame(chart_inner, bg='#1a1f3a')
        self.chart_container.pack(fill=tk.BOTH, expand=True)
        
        self.create_chart()
        
        # ========== STATUS BAR ==========
        status_frame = tk.Frame(self.root, bg='#1a1f3a', height=35)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=(8, 15))
        
        self.status_label = tk.Label(status_frame, text="⚡ Ready to analyze", 
                                     bg='#1a1f3a', fg='#22c55e', 
                                     font=('Segoe UI', 8), anchor='w')
        self.status_label.pack(side=tk.LEFT, padx=12, pady=8)
    
    def create_chart(self):
        """Create chart with proper size"""
        figsize = (18, 10)  # Double the size
        self.figure = Figure(figsize=figsize, facecolor='#0a0e27', dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self.chart_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar for zoom/pan
        toolbar = NavigationToolbar2Tk(self.canvas, self.chart_container)
        toolbar.config(bg='#1a1f3a')
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    def create_prediction_card(self, parent, title, value, color):
        """Create a modern prediction card"""
        card = tk.Frame(parent, bg='#0f1629', relief=tk.FLAT, bd=0)
        
        tk.Label(card, text=title, bg='#0f1629', fg='#94a3b8', 
                font=('Segoe UI', 8)).pack(pady=(10, 3))
        
        value_label = tk.Label(card, text=value, bg='#0f1629', fg=color, 
                              font=('Segoe UI', 16, 'bold'))
        value_label.pack(pady=(0, 10))
        
        card.value_label = value_label
        return card
    
    def enable_all_indicators(self):
        """Enable all indicators"""
        for key in self.indicators:
            self.indicators[key].set(True)
        self.update_chart_only()
    
    def disable_all_indicators(self):
        """Disable all indicators"""
        for key in self.indicators:
            self.indicators[key].set(False)
        self.update_chart_only()
    
    def set_favorite_indicators(self):
        """Set favorite indicators (SMA 20/50, RSI, MACD)"""
        favorites = ['SMA_20', 'SMA_50', 'RSI', 'MACD']
        for key in self.indicators:
            self.indicators[key].set(key in favorites)
        self.update_chart_only()
    
    def update_chart_size(self):
        """Update chart - removed, chart size is now fixed"""
        pass
    
    def toggle_auto_refresh(self):
        """Toggle auto-refresh on/off"""
        if self.auto_refresh.get():
            self.start_auto_refresh()
        else:
            self.stop_auto_refresh()
    
    def start_auto_refresh(self):
        """Start auto-refresh timer"""
        if self.refresh_job:
            self.root.after_cancel(self.refresh_job)
        
        interval_ms = int(self.refresh_interval.get()) * 1000
        self.analyze_market()
        self.refresh_job = self.root.after(interval_ms, self.auto_refresh_callback)
        self.update_status(f"🔄 Auto-refresh enabled ({self.refresh_interval.get()}s)", "success")
    
    def stop_auto_refresh(self):
        """Stop auto-refresh timer"""
        if self.refresh_job:
            self.root.after_cancel(self.refresh_job)
            self.refresh_job = None
        self.update_status("⏸ Auto-refresh disabled", "info")
    
    def auto_refresh_callback(self):
        """Callback for auto-refresh"""
        if self.auto_refresh.get():
            self.analyze_market()
            interval_ms = int(self.refresh_interval.get()) * 1000
            self.refresh_job = self.root.after(interval_ms, self.auto_refresh_callback)
    
    def filter_pairs(self, event=None):
        """Filter pairs based on search"""
        search_term = self.search_var.get().upper()
        
        if not search_term:
            self.pair_combo['values'] = self.filtered_pairs
        else:
            filtered = [p for p in self.filtered_pairs if search_term in p.upper()]
            self.pair_combo['values'] = filtered
            
            if filtered:
                self.pair_combo.set(filtered[0])
    
    def load_trading_pairs(self):
        """Load all trading pairs from Kraken"""
        try:
            self.update_status("Loading trading pairs...", "info")
            
            response = requests.get('https://api.kraken.com/0/public/AssetPairs', timeout=10)
            data = response.json()
            
            if data['error']:
                messagebox.showerror("Error", f"Kraken Error: {data['error']}")
                return
            
            self.all_pairs = data['result']
            
            market_type = self.market_type.get()
            self.filtered_pairs = []
            
            for pair_name, pair_info in self.all_pairs.items():
                if market_type == 'spot':
                    if '.d' not in pair_name and not pair_name.endswith('PERP'):
                        self.filtered_pairs.append(pair_name)
                elif market_type == 'margin':
                    if '.d' in pair_name:
                        self.filtered_pairs.append(pair_name)
                elif market_type == 'futures':
                    if pair_name.endswith('PERP'):
                        self.filtered_pairs.append(pair_name)
            
            self.filtered_pairs.sort()
            self.pair_combo['values'] = self.filtered_pairs
            
            if self.filtered_pairs:
                self.selected_pair.set(self.filtered_pairs[0])
            
            self.update_status(f"✓ {len(self.filtered_pairs)} pairs loaded ({market_type})", "success")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load pairs: {str(e)}")
            self.update_status(f"✗ Error: {str(e)}", "error")
    
    def analyze_market(self):
        """Analyze market with Hopfield Network"""
        thread = threading.Thread(target=self._analyze_market_thread)
        thread.daemon = True
        thread.start()
    
    def _analyze_market_thread(self):
        try:
            self.update_status("Downloading market data...", "info")
            
            pair = self.selected_pair.get()
            interval = self.interval.get()
            
            # Get pattern count from user input
            try:
                pattern_count = int(self.pattern_count.get())
                pattern_count = max(50, min(500, pattern_count))  # Limit between 50-500
            except:
                pattern_count = 100
                self.pattern_count.set('100')
            
            url = f'https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}'
            response = requests.get(url, timeout=15)
            data = response.json()
            
            if data['error']:
                raise Exception(f"Kraken Error: {data['error']}")
            
            pair_key = [k for k in data['result'].keys() if k != 'last'][0]
            ohlc_data = data['result'][pair_key]
            
            # Use pattern_count instead of fixed 200
            self.market_data = []
            for candle in ohlc_data[-pattern_count:]:
                self.market_data.append({
                    'time': datetime.fromtimestamp(candle[0]).strftime('%H:%M'),
                    'timestamp': candle[0],
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[6])
                })
            
            self.update_status("Analyzing patterns...", "info")
            
            self.patterns = self.analyze_patterns(self.market_data)
            
            self.update_status("Training Hopfield Network...", "info")
            pattern_strings = [p['pattern'] for p in self.patterns]
            patterns_trained = self.hopfield_net.train(pattern_strings)
            self.hopfield_trained = True
            
            self.update_status("Generating prediction...", "info")
            self.prediction = self.predict_with_hopfield(self.patterns)
            
            self.root.after(0, self.update_ui)
            
            self.update_status(f"✓ Analysis completed with {patterns_trained} patterns", "success")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.update_status(f"✗ Error: {str(e)}", "error")
    
    def analyze_patterns(self, data):
        """Analyze HH/HL/LH/LL patterns"""
        patterns = []
        
        for i in range(1, len(data)):
            prev = data[i-1]
            curr = data[i]
            
            pattern = ''
            pattern += 'H' if curr['high'] > prev['high'] else 'L'
            pattern += 'H' if curr['low'] > prev['low'] else 'L'
            
            patterns.append({
                'index': i,
                'pattern': pattern,
                'time': curr['time'],
                'close': curr['close'],
                'high': curr['high'],
                'low': curr['low']
            })
        
        return patterns
    
    def predict_with_hopfield(self, patterns):
        """Predict using Hopfield Network"""
        if len(patterns) < 2:
            return None
        
        current_pattern = patterns[-1]['pattern']
        pattern_history = [p['pattern'] for p in patterns]
        
        result = self.hopfield_net.predict_next(current_pattern, pattern_history)
        
        if isinstance(result, tuple):
            predicted, confidence, occurrences = result
        else:
            predicted = result
            confidence = 0
            occurrences = 0
        
        current_binary = self.hopfield_net.pattern_to_binary(current_pattern)
        energy = self.hopfield_net.energy(current_binary)
        
        return {
            'current': current_pattern,
            'predicted': predicted,
            'confidence': confidence,
            'occurrences': occurrences,
            'energy': energy,
            'method': 'Hopfield Network'
        }
    
    def update_chart_only(self):
        """Update only the chart with current data"""
        if self.market_data:
            self.update_graph()
            self.update_status("✓ Chart updated", "success")
    
    def format_price(self, price):
        """Format price based on magnitude"""
        if price >= 1000:
            return f"${price:,.2f}"
        elif price >= 1:
            return f"${price:.4f}"
        elif price >= 0.01:
            return f"${price:.6f}"
        else:
            return f"${price:.8f}"
    
    def update_ui(self):
        """Update interface with results"""
        if self.hopfield_trained:
            self.network_status.config(
                text=f"🟢 Network: Trained with {len(self.patterns)} patterns",
                fg='#22c55e'
            )
        
        if self.prediction:
            self.current_card.value_label.config(
                text=f"{self.prediction['current']} {self.get_pattern_emoji(self.prediction['current'])}"
            )
            self.predicted_card.value_label.config(
                text=f"{self.prediction['predicted']} {self.get_pattern_emoji(self.prediction['predicted'])}"
            )
            self.confidence_card.value_label.config(
                text=f"{self.prediction['confidence']:.1f}%"
            )
            self.energy_card.value_label.config(
                text=f"{self.prediction['energy']:.2f}"
            )
        
        self.update_graph()
    
    def update_graph(self):
        """Update price chart with technical indicators"""
        self.figure.clear()
        
        # Determine number of subplots needed
        num_subplots = 1
        if self.indicators['RSI'].get():
            num_subplots += 1
        if self.indicators['MACD'].get():
            num_subplots += 1
        if self.indicators['Stochastic'].get():
            num_subplots += 1
        if self.indicators['Volume'].get():
            num_subplots += 1
        if self.indicators['ATR'].get():
            num_subplots += 1
        if self.indicators['CCI'].get():
            num_subplots += 1
        if self.indicators['ADX'].get():
            num_subplots += 1
        if self.indicators['OBV'].get():
            num_subplots += 1
        if self.indicators['WilliamsR'].get():
            num_subplots += 1
        
        # Create subplots with custom height ratios
        if num_subplots == 1:
            height_ratios = [1]
        elif num_subplots == 2:
            height_ratios = [3, 1]
        elif num_subplots == 3:
            height_ratios = [3, 1, 1]
        else:
            height_ratios = [3] + [1] * (num_subplots - 1)
        
        gs = self.figure.add_gridspec(num_subplots, 1, height_ratios=height_ratios, hspace=0.3)
        
        # Main price chart
        ax_price = self.figure.add_subplot(gs[0], facecolor='#0a0e27')
        
        data_to_plot = self.market_data[-100:]
        
        times = [d['time'] for d in data_to_plot]
        highs = [d['high'] for d in data_to_plot]
        lows = [d['low'] for d in data_to_plot]
        closes = [d['close'] for d in data_to_plot]
        opens = [d['open'] for d in data_to_plot]
        volumes = [d['volume'] for d in data_to_plot]
        
        x_indices = list(range(len(times)))
        
        # Candlestick chart
        for i in x_indices:
            color = '#22c55e' if closes[i] >= opens[i] else '#ef4444'
            ax_price.plot([i, i], [lows[i], highs[i]], color=color, linewidth=1, alpha=0.8)
            
            body_height = abs(closes[i] - opens[i])
            body_bottom = min(opens[i], closes[i])
            
            rect = mpatches.Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                                     facecolor=color, edgecolor=color, alpha=0.9)
            ax_price.add_patch(rect)
        
        # Technical Indicators on price chart
        legend_elements = []
        
        if self.indicators['SMA_20'].get():
            sma_20 = TechnicalIndicators.sma(closes, 20)
            valid_indices = [i for i, v in enumerate(sma_20) if v is not None]
            valid_values = [v for v in sma_20 if v is not None]
            line, = ax_price.plot(valid_indices, valid_values, color='#fbbf24', 
                                 linewidth=2.5, label='SMA 20', alpha=0.9)
            legend_elements.append(line)
        
        if self.indicators['SMA_50'].get():
            sma_50 = TechnicalIndicators.sma(closes, 50)
            valid_indices = [i for i, v in enumerate(sma_50) if v is not None]
            valid_values = [v for v in sma_50 if v is not None]
            line, = ax_price.plot(valid_indices, valid_values, color='#f59e0b', 
                                 linewidth=2.5, label='SMA 50', alpha=0.9)
            legend_elements.append(line)
        
        if self.indicators['EMA_12'].get():
            ema_12 = TechnicalIndicators.ema(closes, 12)
            valid_indices = [i for i, v in enumerate(ema_12) if v is not None]
            valid_values = [v for v in ema_12 if v is not None]
            line, = ax_price.plot(valid_indices, valid_values, color='#10b981', 
                                 linewidth=2.5, label='EMA 12', alpha=0.9, linestyle='--')
            legend_elements.append(line)
        
        if self.indicators['EMA_26'].get():
            ema_26 = TechnicalIndicators.ema(closes, 26)
            valid_indices = [i for i, v in enumerate(ema_26) if v is not None]
            valid_values = [v for v in ema_26 if v is not None]
            line, = ax_price.plot(valid_indices, valid_values, color='#059669', 
                                 linewidth=2.5, label='EMA 26', alpha=0.9, linestyle='--')
            legend_elements.append(line)
        
        if self.indicators['EMA_50'].get():
            ema_50 = TechnicalIndicators.ema(closes, 50)
            valid_indices = [i for i, v in enumerate(ema_50) if v is not None]
            valid_values = [v for v in ema_50 if v is not None]
            line, = ax_price.plot(valid_indices, valid_values, color='#047857', 
                                 linewidth=2.5, label='EMA 50', alpha=0.9, linestyle='--')
            legend_elements.append(line)
        
        if self.indicators['SMA_200'].get():
            sma_200 = TechnicalIndicators.sma(closes, 200)
            valid_indices = [i for i, v in enumerate(sma_200) if v is not None]
            valid_values = [v for v in sma_200 if v is not None]
            if valid_indices:
                line, = ax_price.plot(valid_indices, valid_values, color='#ea580c', 
                                     linewidth=3, label='SMA 200', alpha=0.9)
                legend_elements.append(line)
        
        if self.indicators['BB'].get():
            sma, upper, lower = TechnicalIndicators.bollinger_bands(closes, 20, 2)
            valid_indices = [i for i, v in enumerate(upper) if v is not None]
            valid_upper = [v for v in upper if v is not None]
            valid_lower = [lower[i] for i in valid_indices]
            
            ax_price.plot(valid_indices, valid_upper, color='#8b5cf6', 
                         linewidth=1.5, label='BB Upper', alpha=0.7, linestyle=':')
            ax_price.plot(valid_indices, valid_lower, color='#8b5cf6', 
                         linewidth=1.5, label='BB Lower', alpha=0.7, linestyle=':')
            ax_price.fill_between(valid_indices, valid_lower, valid_upper, 
                                 color='#8b5cf6', alpha=0.1)
            
            legend_elements.append(mpatches.Patch(color='#8b5cf6', alpha=0.3, label='Bollinger Bands'))
        
        if self.indicators['Ichimoku'].get():
            ichimoku = TechnicalIndicators.ichimoku(closes, 9, 26, 52, 26)
            
            # Tenkan-sen (Conversion Line) - Red
            tenkan = ichimoku['tenkan_sen']
            valid_idx = [i for i, v in enumerate(tenkan) if v is not None]
            if valid_idx:
                valid_vals = [v for v in tenkan if v is not None]
                line, = ax_price.plot(valid_idx, valid_vals, color='#ef4444', 
                                     linewidth=2, label='Tenkan-sen', alpha=0.8)
                legend_elements.append(line)
            
            # Kijun-sen (Base Line) - Blue
            kijun = ichimoku['kijun_sen']
            valid_idx = [i for i, v in enumerate(kijun) if v is not None]
            if valid_idx:
                valid_vals = [v for v in kijun if v is not None]
                line, = ax_price.plot(valid_idx, valid_vals, color='#3b82f6', 
                                     linewidth=2, label='Kijun-sen', alpha=0.8)
                legend_elements.append(line)
            
            # Senkou Span A & B (Cloud)
            senkou_a = ichimoku['senkou_span_a']
            senkou_b = ichimoku['senkou_span_b']
            
            valid_idx_a = [i for i, v in enumerate(senkou_a) if v is not None and i < len(senkou_b) and senkou_b[i] is not None]
            if valid_idx_a:
                valid_a = [senkou_a[i] for i in valid_idx_a]
                valid_b = [senkou_b[i] for i in valid_idx_a]
                
                ax_price.plot(valid_idx_a, valid_a, color='#06b6d4', 
                             linewidth=1, alpha=0.5, linestyle='--')
                ax_price.plot(valid_idx_a, valid_b, color='#f97316', 
                             linewidth=1, alpha=0.5, linestyle='--')
                
                # Fill cloud with different colors based on which span is higher
                for i in range(len(valid_idx_a) - 1):
                    idx_start = valid_idx_a[i]
                    idx_end = valid_idx_a[i + 1]
                    if valid_a[i] > valid_b[i]:
                        # Bullish cloud (green)
                        ax_price.fill_between([idx_start, idx_end], 
                                             [valid_b[i], valid_b[i+1]], 
                                             [valid_a[i], valid_a[i+1]], 
                                             color='#22c55e', alpha=0.1)
                    else:
                        # Bearish cloud (red)
                        ax_price.fill_between([idx_start, idx_end], 
                                             [valid_a[i], valid_a[i+1]], 
                                             [valid_b[i], valid_b[i+1]], 
                                             color='#ef4444', alpha=0.1)
                
                legend_elements.append(mpatches.Patch(color='#06b6d4', alpha=0.3, label='Ichimoku Cloud'))
        
        ax_price.set_xlabel('Time', color='#e2e8f0', fontsize=12, fontweight='bold')
        ax_price.set_ylabel('Price (USD)', color='#e2e8f0', fontsize=12, fontweight='bold')
        ax_price.set_title(f'{self.selected_pair.get()} - Technical Analysis', 
                          color='#e2e8f0', fontsize=15, fontweight='bold', pad=20)
        
        step = max(1, len(times) // 12)
        ax_price.set_xticks(x_indices[::step])
        ax_price.set_xticklabels(times[::step], rotation=45, ha='right', fontsize=9, color='#94a3b8')
        
        def price_formatter(x, p):
            if x >= 1000:
                return f'${x:,.0f}'
            elif x >= 1:
                return f'${x:.2f}'
            elif x >= 0.01:
                return f'${x:.4f}'
            else:
                return f'${x:.6f}'
        
        ax_price.yaxis.set_major_formatter(FuncFormatter(price_formatter))
        ax_price.tick_params(colors='#94a3b8', labelsize=10, which='both')
        
        if legend_elements:
            ax_price.legend(handles=legend_elements, facecolor='#1a1f3a', edgecolor='#3b82f6', 
                          labelcolor='#e2e8f0', fontsize=10, loc='upper left', framealpha=0.9)
        
        ax_price.grid(True, alpha=0.15, color='#475569', linestyle='--', linewidth=0.5)
        
        for spine in ax_price.spines.values():
            spine.set_color('#475569')
        
        # Additional subplots
        subplot_idx = 1
        
        # RSI
        if self.indicators['RSI'].get():
            ax_rsi = self.figure.add_subplot(gs[subplot_idx], facecolor='#0a0e27')
            rsi = TechnicalIndicators.rsi(closes, 14)
            valid_indices = [i for i, v in enumerate(rsi) if v is not None]
            valid_values = [v for v in rsi if v is not None]
            
            ax_rsi.plot(valid_indices, valid_values, color='#ef4444', linewidth=2, label='RSI (14)')
            ax_rsi.axhline(y=70, color='#f87171', linestyle='--', linewidth=1, alpha=0.7, label='Overbought')
            ax_rsi.axhline(y=30, color='#22c55e', linestyle='--', linewidth=1, alpha=0.7, label='Oversold')
            ax_rsi.fill_between(valid_indices, 30, 70, color='#475569', alpha=0.1)
            
            ax_rsi.set_ylabel('RSI', color='#e2e8f0', fontsize=11, fontweight='bold')
            ax_rsi.set_ylim(0, 100)
            ax_rsi.tick_params(colors='#94a3b8', labelsize=9)
            ax_rsi.legend(facecolor='#1a1f3a', edgecolor='#ef4444', labelcolor='#e2e8f0', 
                         fontsize=9, loc='upper left', framealpha=0.9)
            ax_rsi.grid(True, alpha=0.15, color='#475569', linestyle='--', linewidth=0.5)
            
            for spine in ax_rsi.spines.values():
                spine.set_color('#475569')
            
            subplot_idx += 1
        
        # Stochastic
        if self.indicators['Stochastic'].get():
            ax_stoch = self.figure.add_subplot(gs[subplot_idx], facecolor='#0a0e27')
            k_values, d_values = TechnicalIndicators.stochastic(highs, lows, closes, 14, 3)
            
            valid_k_idx = [i for i, v in enumerate(k_values) if v is not None]
            if valid_k_idx:
                valid_k = [k_values[i] for i in valid_k_idx]
                ax_stoch.plot(valid_k_idx, valid_k, color='#ec4899', 
                             linewidth=2, label='%K', alpha=0.9)
            
            valid_d_idx = [i for i, v in enumerate(d_values) if v is not None]
            if valid_d_idx:
                valid_d = [d_values[i] for i in valid_d_idx]
                ax_stoch.plot(valid_d_idx, valid_d, color='#f59e0b', 
                             linewidth=2, label='%D', alpha=0.9)
            
            ax_stoch.axhline(y=80, color='#f87171', linestyle='--', linewidth=1, alpha=0.7)
            ax_stoch.axhline(y=20, color='#22c55e', linestyle='--', linewidth=1, alpha=0.7)
            ax_stoch.fill_between(range(len(closes)), 20, 80, color='#475569', alpha=0.1)
            
            ax_stoch.set_ylabel('Stochastic', color='#e2e8f0', fontsize=11, fontweight='bold')
            ax_stoch.set_ylim(0, 100)
            ax_stoch.tick_params(colors='#94a3b8', labelsize=9)
            ax_stoch.legend(facecolor='#1a1f3a', edgecolor='#ec4899', labelcolor='#e2e8f0', 
                          fontsize=9, loc='upper left', framealpha=0.9)
            ax_stoch.grid(True, alpha=0.15, color='#475569', linestyle='--', linewidth=0.5)
            
            for spine in ax_stoch.spines.values():
                spine.set_color('#475569')
            
            subplot_idx += 1
        
        # ATR
        if self.indicators['ATR'].get():
            ax_atr = self.figure.add_subplot(gs[subplot_idx], facecolor='#0a0e27')
            atr = TechnicalIndicators.atr(highs, lows, closes, 14)
            
            valid_idx = [i for i, v in enumerate(atr) if v is not None]
            if valid_idx:
                valid_vals = [atr[i] for i in valid_idx]
                ax_atr.plot(valid_idx, valid_vals, color='#14b8a6', 
                           linewidth=2, label='ATR (14)', alpha=0.9)
                ax_atr.fill_between(valid_idx, 0, valid_vals, color='#14b8a6', alpha=0.2)
            
            ax_atr.set_ylabel('ATR', color='#e2e8f0', fontsize=11, fontweight='bold')
            ax_atr.tick_params(colors='#94a3b8', labelsize=9)
            ax_atr.legend(facecolor='#1a1f3a', edgecolor='#14b8a6', labelcolor='#e2e8f0', 
                         fontsize=9, loc='upper left', framealpha=0.9)
            ax_atr.grid(True, alpha=0.15, color='#475569', linestyle='--', linewidth=0.5)
            
            for spine in ax_atr.spines.values():
                spine.set_color('#475569')
            
            subplot_idx += 1
        
        # CCI
        if self.indicators['CCI'].get():
            ax_cci = self.figure.add_subplot(gs[subplot_idx], facecolor='#0a0e27')
            cci = TechnicalIndicators.cci(highs, lows, closes, 20)
            
            valid_idx = [i for i, v in enumerate(cci) if v is not None]
            if valid_idx:
                valid_vals = [cci[i] for i in valid_idx]
                ax_cci.plot(valid_idx, valid_vals, color='#f97316', 
                           linewidth=2, label='CCI (20)', alpha=0.9)
            
            ax_cci.axhline(y=100, color='#f87171', linestyle='--', linewidth=1, alpha=0.7)
            ax_cci.axhline(y=-100, color='#22c55e', linestyle='--', linewidth=1, alpha=0.7)
            ax_cci.axhline(y=0, color='#94a3b8', linestyle='-', linewidth=1, alpha=0.5)
            
            ax_cci.set_ylabel('CCI', color='#e2e8f0', fontsize=11, fontweight='bold')
            ax_cci.tick_params(colors='#94a3b8', labelsize=9)
            ax_cci.legend(facecolor='#1a1f3a', edgecolor='#f97316', labelcolor='#e2e8f0', 
                         fontsize=9, loc='upper left', framealpha=0.9)
            ax_cci.grid(True, alpha=0.15, color='#475569', linestyle='--', linewidth=0.5)
            
            for spine in ax_cci.spines.values():
                spine.set_color('#475569')
            
            subplot_idx += 1
        
        # ADX
        if self.indicators['ADX'].get():
            ax_adx = self.figure.add_subplot(gs[subplot_idx], facecolor='#0a0e27')
            adx, plus_di, minus_di = TechnicalIndicators.adx(highs, lows, closes, 14)
            
            valid_adx_idx = [i for i, v in enumerate(adx) if v is not None]
            if valid_adx_idx:
                valid_adx = [adx[i] for i in valid_adx_idx]
                ax_adx.plot(valid_adx_idx, valid_adx, color='#84cc16', 
                           linewidth=2.5, label='ADX', alpha=0.9)
            
            valid_plus_idx = [i for i, v in enumerate(plus_di) if v is not None]
            if valid_plus_idx:
                valid_plus = [plus_di[i] for i in valid_plus_idx]
                ax_adx.plot(valid_plus_idx, valid_plus, color='#22c55e', 
                           linewidth=1.5, label='+DI', alpha=0.7, linestyle='--')
            
            valid_minus_idx = [i for i, v in enumerate(minus_di) if v is not None]
            if valid_minus_idx:
                valid_minus = [minus_di[i] for i in valid_minus_idx]
                ax_adx.plot(valid_minus_idx, valid_minus, color='#ef4444', 
                           linewidth=1.5, label='-DI', alpha=0.7, linestyle='--')
            
            ax_adx.axhline(y=25, color='#fbbf24', linestyle='--', linewidth=1, alpha=0.7)
            
            ax_adx.set_ylabel('ADX', color='#e2e8f0', fontsize=11, fontweight='bold')
            ax_adx.tick_params(colors='#94a3b8', labelsize=9)
            ax_adx.legend(facecolor='#1a1f3a', edgecolor='#84cc16', labelcolor='#e2e8f0', 
                         fontsize=9, loc='upper left', framealpha=0.9)
            ax_adx.grid(True, alpha=0.15, color='#475569', linestyle='--', linewidth=0.5)
            
            for spine in ax_adx.spines.values():
                spine.set_color('#475569')
            
            subplot_idx += 1
        
        # OBV
        if self.indicators['OBV'].get():
            ax_obv = self.figure.add_subplot(gs[subplot_idx], facecolor='#0a0e27')
            obv = TechnicalIndicators.obv(closes, volumes)
            
            ax_obv.plot(range(len(obv)), obv, color='#a855f7', 
                       linewidth=2, label='OBV', alpha=0.9)
            ax_obv.fill_between(range(len(obv)), 0, obv, color='#a855f7', alpha=0.2)
            
            ax_obv.set_ylabel('OBV', color='#e2e8f0', fontsize=11, fontweight='bold')
            ax_obv.tick_params(colors='#94a3b8', labelsize=9)
            ax_obv.legend(facecolor='#1a1f3a', edgecolor='#a855f7', labelcolor='#e2e8f0', 
                         fontsize=9, loc='upper left', framealpha=0.9)
            ax_obv.grid(True, alpha=0.15, color='#475569', linestyle='--', linewidth=0.5)
            
            for spine in ax_obv.spines.values():
                spine.set_color('#475569')
            
            subplot_idx += 1
        
        # Williams %R
        if self.indicators['WilliamsR'].get():
            ax_wr = self.figure.add_subplot(gs[subplot_idx], facecolor='#0a0e27')
            wr = TechnicalIndicators.williams_r(highs, lows, closes, 14)
            
            valid_idx = [i for i, v in enumerate(wr) if v is not None]
            if valid_idx:
                valid_vals = [wr[i] for i in valid_idx]
                ax_wr.plot(valid_idx, valid_vals, color='#f43f5e', 
                          linewidth=2, label='Williams %R', alpha=0.9)
            
            ax_wr.axhline(y=-20, color='#f87171', linestyle='--', linewidth=1, alpha=0.7)
            ax_wr.axhline(y=-80, color='#22c55e', linestyle='--', linewidth=1, alpha=0.7)
            ax_wr.fill_between(range(len(closes)), -20, -80, color='#475569', alpha=0.1)
            
            ax_wr.set_ylabel('Williams %R', color='#e2e8f0', fontsize=11, fontweight='bold')
            ax_wr.set_ylim(-100, 0)
            ax_wr.tick_params(colors='#94a3b8', labelsize=9)
            ax_wr.legend(facecolor='#1a1f3a', edgecolor='#f43f5e', labelcolor='#e2e8f0', 
                        fontsize=9, loc='upper left', framealpha=0.9)
            ax_wr.grid(True, alpha=0.15, color='#475569', linestyle='--', linewidth=0.5)
            
            for spine in ax_wr.spines.values():
                spine.set_color('#475569')
            
            subplot_idx += 1
        
        # MACD
        if self.indicators['MACD'].get():
            ax_macd = self.figure.add_subplot(gs[subplot_idx], facecolor='#0a0e27')
            macd_line, signal_line, histogram = TechnicalIndicators.macd(closes, 12, 26, 9)
            
            # Plot MACD line
            valid_macd_idx = [i for i, v in enumerate(macd_line) if v is not None]
            if valid_macd_idx:
                valid_macd = [macd_line[i] for i in valid_macd_idx]
                ax_macd.plot(valid_macd_idx, valid_macd, color='#3b82f6', 
                            linewidth=2.5, label='MACD', alpha=0.9)
            
            # Plot Signal line
            valid_signal_idx = [i for i, v in enumerate(signal_line) if v is not None]
            if valid_signal_idx:
                valid_signal = [signal_line[i] for i in valid_signal_idx]
                ax_macd.plot(valid_signal_idx, valid_signal, color='#f59e0b', 
                            linewidth=2.5, label='Signal', alpha=0.9)
            
            # Plot Histogram
            valid_hist_idx = [i for i, v in enumerate(histogram) if v is not None]
            if valid_hist_idx:
                valid_hist = [histogram[i] for i in valid_hist_idx]
                colors = ['#22c55e' if v >= 0 else '#ef4444' for v in valid_hist]
                ax_macd.bar(valid_hist_idx, valid_hist, color=colors, alpha=0.6, width=0.8)
            
            ax_macd.axhline(y=0, color='#94a3b8', linestyle='-', linewidth=1, alpha=0.5)
            ax_macd.set_ylabel('MACD', color='#e2e8f0', fontsize=11, fontweight='bold')
            ax_macd.tick_params(colors='#94a3b8', labelsize=9)
            ax_macd.legend(facecolor='#1a1f3a', edgecolor='#3b82f6', labelcolor='#e2e8f0', 
                          fontsize=9, loc='upper left', framealpha=0.9)
            ax_macd.grid(True, alpha=0.15, color='#475569', linestyle='--', linewidth=0.5)
            
            for spine in ax_macd.spines.values():
                spine.set_color('#475569')
            
            subplot_idx += 1
        
        # Volume
        if self.indicators['Volume'].get():
            ax_vol = self.figure.add_subplot(gs[subplot_idx], facecolor='#0a0e27')
            colors_vol = ['#22c55e' if closes[i] >= opens[i] else '#ef4444' for i in x_indices]
            ax_vol.bar(x_indices, volumes, color=colors_vol, alpha=0.6, width=0.8)
            
            ax_vol.set_ylabel('Volume', color='#e2e8f0', fontsize=11, fontweight='bold')
            ax_vol.tick_params(colors='#94a3b8', labelsize=9)
            ax_vol.grid(True, alpha=0.15, color='#475569', linestyle='--', linewidth=0.5)
            
            # Format volume axis
            ax_vol.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.1f}K'))
            
            for spine in ax_vol.spines.values():
                spine.set_color('#475569')
            
            subplot_idx += 1
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def get_pattern_emoji(self, pattern):
        """Get emoji for pattern"""
        if pattern == 'HH': return '🚀'
        elif pattern == 'HL': return '📈'
        elif pattern == 'LH': return '📊'
        elif pattern == 'LL': return '📉'
        return '❓'
    
    def export_data(self):
        """Export analysis to JSON"""
        try:
            active_indicators = {k: v.get() for k, v in self.indicators.items()}
            
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'pair': self.selected_pair.get(),
                'interval': self.interval.get(),
                'market_type': self.market_type.get(),
                'active_indicators': active_indicators,
                'prediction': self.prediction,
                'hopfield_network': {
                    'trained': self.hopfield_trained,
                    'patterns_learned': len(self.patterns),
                    'weights_matrix': self.hopfield_net.weights.tolist() if self.hopfield_trained else None
                },
                'patterns': self.patterns[-50:],
                'market_data': self.market_data[-50:]
            }
            
            filename = f"hopfield_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            messagebox.showinfo("Success", f"Analysis exported to: {filename}")
            self.update_status(f"✓ Exported: {filename}", "success")
            
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    def update_status(self, message, status_type="info"):
        """Update status bar"""
        colors = {
            'info': '#60a5fa',
            'success': '#22c55e',
            'error': '#ef4444'
        }
        self.status_label.config(text=message, fg=colors.get(status_type, '#60a5fa'))

if __name__ == "__main__":
    root = tk.Tk()
    app = KrakenPredictorApp(root)
    root.mainloop()