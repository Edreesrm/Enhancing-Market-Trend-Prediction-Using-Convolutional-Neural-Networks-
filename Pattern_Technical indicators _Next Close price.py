# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 12:27:20 2024

@author: edree
"""

import talib
import pandas as pd
import mplfinance as mpf
import os
import matplotlib.pyplot as plt

dd = 'D:/All Data/model/Data/EURUSD_M15.csv'
data = pd.read_csv(dd, delimiter='\t', index_col='Time', parse_dates=True)

data['SMA'] = talib.SMA(data['Close'], timeperiod=20)

pattern_funcs = [
    ("Two Crows", talib.CDL2CROWS),
    ("Three Black Crows", talib.CDL3BLACKCROWS),
    ("Three Inside Up/Down", talib.CDL3INSIDE),
    ("Three-Line Strike", talib.CDL3LINESTRIKE),
    ("Three Outside Up/Down", talib.CDL3OUTSIDE),
    ("Three Stars In The South", talib.CDL3STARSINSOUTH),
    ("Three Advancing White Soldiers", talib.CDL3WHITESOLDIERS),
    ("Abandoned Baby", talib.CDLABANDONEDBABY),
    ("Advance Block", talib.CDLADVANCEBLOCK),
    ("Belt-hold", talib.CDLBELTHOLD),
    ("Breakaway", talib.CDLBREAKAWAY),
    ("Closing Marubozu", talib.CDLCLOSINGMARUBOZU),
    ("Concealing Baby Swallow", talib.CDLCONCEALBABYSWALL),
    ("Counterattack", talib.CDLCOUNTERATTACK),
    ("Dark Cloud Cover", talib.CDLDARKCLOUDCOVER),
    ("Doji", talib.CDLDOJI),
    ("Doji Star", talib.CDLDOJISTAR),
    ("Dragonfly Doji", talib.CDLDRAGONFLYDOJI),
    ("Engulfing Pattern", talib.CDLENGULFING),
    ("Evening Doji Star", talib.CDLEVENINGDOJISTAR),
    ("Evening Star", talib.CDLEVENINGSTAR),
    ("Up/Down-gap side-by-side white lines", talib.CDLGAPSIDESIDEWHITE),
    ("Gravestone Doji", talib.CDLGRAVESTONEDOJI),
    ("Hammer", talib.CDLHAMMER),
    ("Hanging Man", talib.CDLHANGINGMAN),
    ("Harami Pattern", talib.CDLHARAMI),
    ("Harami Cross Pattern", talib.CDLHARAMICROSS),
    ("High-Wave Candle", talib.CDLHIGHWAVE),
    ("Hikkake Pattern", talib.CDLHIKKAKE),
    ("Modified Hikkake Pattern", talib.CDLHIKKAKEMOD),
    ("Homing Pigeon", talib.CDLHOMINGPIGEON),
    ("Identical Three Crows", talib.CDLIDENTICAL3CROWS),
    ("In-Neck Pattern", talib.CDLINNECK),
    ("Inverted Hammer", talib.CDLINVERTEDHAMMER),
    ("Kicking", talib.CDLKICKING),
    ("Kicking - bull/bear determined by the longer marubozu", talib.CDLKICKINGBYLENGTH),
    ("Ladder Bottom", talib.CDLLADDERBOTTOM),
    ("Long Legged Doji", talib.CDLLONGLEGGEDDOJI),
    ("Long Line Candle", talib.CDLLONGLINE),
    ("Marubozu", talib.CDLMARUBOZU),
    ("Matching Low", talib.CDLMATCHINGLOW),
    ("Mat Hold", talib.CDLMATHOLD),
    ("Morning Doji Star", talib.CDLMORNINGDOJISTAR),
    ("Morning Star", talib.CDLMORNINGSTAR),
    ("On-Neck Pattern", talib.CDLONNECK),
    ("Piercing Pattern", talib.CDLPIERCING),
    ("Rickshaw Man", talib.CDLRICKSHAWMAN),
    ("Rising/Falling Three Methods", talib.CDLRISEFALL3METHODS),
    ("Separating Lines", talib.CDLSEPARATINGLINES),
    ("Shooting Star", talib.CDLSHOOTINGSTAR),
    ("Short Line Candle", talib.CDLSHORTLINE),
    ("Spinning Top", talib.CDLSPINNINGTOP),
    ("Stalled Pattern", talib.CDLSTALLEDPATTERN),
    ("Stick Sandwich", talib.CDLSTICKSANDWICH),
    ("Takuri (Dragonfly Doji with very long lower shadow)", talib.CDLTAKURI),
    ("Tasuki Gap", talib.CDLTASUKIGAP),
    ("Thrusting Pattern", talib.CDLTHRUSTING),
    ("Tristar Pattern", talib.CDLTRISTAR),
    ("Unique 3 River", talib.CDLUNIQUE3RIVER),
    ("Upside Gap Two Crows", talib.CDLUPSIDEGAP2CROWS),
    ("Upside/Downside Gap Three Methods", talib.CDLXSIDEGAP3METHODS)
]

output_dir = "chart_images5_1"
os.makedirs(output_dir, exist_ok=True)
uptrend_dir = os.path.join(output_dir, "uptrend")
downtrend_dir = os.path.join(output_dir, "downtrend")
os.makedirs(uptrend_dir, exist_ok=True)
os.makedirs(downtrend_dir, exist_ok=True)
window_size=5
shift_size=1
for i in range(0, len(data) - window_size,shift_size):
    window = data.iloc[i:i+window_size]
    next_candle_close = data['Close'].iloc[i+window_size]
    last_candle_close = window['Close'].iloc[-1]
    sma_last = window['SMA'].iloc[-1]

    pattern_detected = False

    for name, func in pattern_funcs:
        if func(window['Open'], window['High'], window['Low'], window['Close']).iloc[-1] != 0:
            pattern_detected = True
            break


    if pattern_detected:
        if last_candle_close > sma_last :
            label = 'uptrend'
            save_path = os.path.join(uptrend_dir, f"{label}_{i}.png")
        elif last_candle_close < sma_last:
            label = 'downtrend'
            save_path = os.path.join(downtrend_dir, f"{label}_{i}.png")
        else:
            continue

        ap = [mpf.make_addplot(window['SMA'], color='blue', secondary_y=False)]
        mpf.plot(window, type='candle', style='yahoo', addplot=ap, volume=True, axisoff=True, ylabel='',
 savefig=save_path)
        plt.close()

