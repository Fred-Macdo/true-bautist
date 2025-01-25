# Technical Indicators
def calculate_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Simple Moving Average"""
    return df['close'].rolling(window=period).mean()

def calculate_ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return df['close'].ewm(span=period, adjust=False).mean()

def calculate_wma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Weighted Moving Average"""
    weights = np.arange(1, period + 1)
    return df['close'].rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum())

def calculate_hull_ma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate Hull Moving Average
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    """
    half_period = int(period/2)
    sqrt_period = int(np.sqrt(period))
    
    wma1 = calculate_wma(df, half_period)
    wma2 = calculate_wma(df, period)
    return calculate_wma(pd.DataFrame({'close': 2*wma1 - wma2}), sqrt_period)

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD, Signal line, and MACD histogram"""
    exp1 = df['close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands"""
    middle = df['close'].rolling(window=period).mean()
    std_dev = df['close'].rolling(window=period).std()
    upper = middle + std * std_dev
    lower = middle - std * std_dev
    return upper, middle, lower

def calculate_keltner_channels(df: pd.DataFrame, period: int = 20, atr_mult: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Keltner Channels"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    middle = typical_price.rolling(window=period).mean()
    atr = calculate_atr(df, period)
    
    upper = middle + (atr_mult * atr)
    lower = middle - (atr_mult * atr)
    return upper, middle, lower

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close'].shift()
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    return tr.rolling(window=period).mean()

def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Calculate Stochastic Oscillator"""
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    k = 100 * (df['close'] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period).mean()
    return k, d

def calculate_roc(df: pd.DataFrame, period: int = 12) -> pd.Series:
    """Calculate Rate of Change"""
    return (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100

def calculate_momentum(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Momentum"""
    return df['close'] - df['close'].shift(period)

def calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Williams %R"""
    highest_high = df['high'].rolling(window=period).max()
    lowest_low = df['low'].rolling(window=period).min()
    wr = (highest_high - df['close']) / (highest_high - lowest_low) * -100
    return wr

def calculate_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Average Directional Index (ADX)"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # Calculate Plus and Minus Directional Movement
    up_move = high - high.shift()
    down_move = low.shift() - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Calculate Plus and Minus Directional Indicators
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
    
    # Calculate ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx, plus_di, minus_di

def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Commodity Channel Index"""
    tp = (df['high'] + df['low'] + df['close']) / 3
    tp_sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: pd.Series(x).mad())
    cci = (tp - tp_sma) / (0.015 * mad)
    return cci

def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """Calculate On-Balance Volume"""
    obv = pd.Series(0, index=df.index)
    obv.iloc[0] = df['volume'].iloc[0]
    
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate Volume Weighted Average Price"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

def calculate_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Money Flow Index"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    
    positive_flow = pd.Series(0, index=df.index)
    negative_flow = pd.Series(0, index=df.index)
    
    # Calculate positive and negative money flow
    for i in range(1, len(df)):
        if typical_price[i] > typical_price[i-1]:
            positive_flow[i] = money_flow[i]
        else:
            negative_flow[i] = money_flow[i]
    
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    
    mfi = 100 - (100 / (1 + positive_mf / negative_mf))
    return mfi

def calculate_fibonacci_levels(df: pd.DataFrame, period: int = 20) -> Dict[str, pd.Series]:
    """Calculate Fibonacci Retracement Levels"""
    high = df['high'].rolling(window=period).max()
    low = df['low'].rolling(window=period).min()
    diff = high - low
    
    return {
        'level_0': high,
        'level_236': high - diff * 0.236,
        'level_382': high - diff * 0.382,
        'level_500': high - diff * 0.500,
        'level_618': high - diff * 0.618,
        'level_100': low
    }
