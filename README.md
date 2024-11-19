This repository contains a collection of Python scripts that aim to predict the direction of EUR/USD forex prices for the 15-minute time frame using deep learning techniques and technical indicators.

**Files**
1. CNN.py
   
    Contains code to train a Convolutional Neural Network (CNN) model for classifying price patterns as uptrends or downtrends.
  
2. complexity.py
   
    Implements time complexity calculations for CNN model operations.

3. EURUSD_M15.csv

    A dataset of EUR/USD prices on a 15-minute timeframe with columns for open, close, high, low prices, and volume.

4. Pattern_Technical indicators _Next Close price.py
   
    Uses the TA-Lib library to extract candlestick patterns and create sub-candlestick charts using sliding window techniques. sub-candlestick charts are classified into binary classes (uptrend and downtrend)

    based on signals from the indicator (SMA).

5. Test for Predictions.py
   
    Creates a GUI using Tkinter for real-time predictions on buy or sell decisions based on the trained model.

6. Cross-Validation.py
   
   The cross-validation techniques evaluate model performance on multiple data splits, ensuring robust and unbiased results. It also helps assess the predictive model's generalization capability.
   
7. Complexity Analysis.py
   
   Analyzes the complexity of a CNN model by calculating total parameters, FLOPs (floating-point operations), memory usage, and inference time. 

**Libraries required for installation are** 

    pip install numpy pandas matplotlib scikit-learn tensorflow TA-Lib pillow

**Usage**
1. **Load the Dataset**
   
     Ensure EURUSD_M15.csv is placed in the project directory.

2. **Pattern Detection and Technical Indicators**

      Run Pattern_Technical_indicators_Next_Close_price.py to:
   
     Extract technical indicators using the TA-Lib library.
   
     **Generate Sub-Candlestick Charts:**
   
     Create sub-candlestick charts using sliding window techniques.
   
     **Window Configuration:**
   
     **Window Size:** You can choose between a fixed window size or select a custom size based on your analysis requirements.
   
     **Shift Size:** Similar to window size, the shift (stride) can be fixed or adjustable to control the overlap between consecutive windows.
   
     Classify patterns into binary classes (uptrend and downtrend) based on SMA indicator signals.
   
     The output predictions for the next candlestick will be saved in two folders:
   
     uptrend_sub_charts
   
     downtrend_sub_charts

4. **Train the CNN Model**
   
     Execute CNN.py to train the Convolutional Neural Network model using the generated sub-charts.
   
5. **Predictions with GUI**
   
     Launch the GUI by running Tkinter_Predictions.py to receive real-time buy or sell predictions based on the trained model.

**Future Work**

**Enhance Ensemble Labeling with Additional Indicators:** 
   
   Integrate more technical indicators and refine the ensemble labeling method to improve signal accuracy and reliability.

**Apply Wavelet Transform for Noise Reduction:**
   
   Use wavelet transforms to preprocess data, isolating trends and reducing noise, thereby improving model inputs for more accurate predictions.
