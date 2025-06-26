################################ GLOBAL VARIABLES ################################

class global_variables:

    # Main
    CRYPTO = 'BTCUSDT'  # Binance symbol for BTC/USDT
    NEURAL_NETWORK = 'may_gru_100.keras'  # Neural Network model file
    TRAINING_DATA = 'train_may_june_01_11.csv'   # CSV where historical and live prices are stored
    TEST_DATA = ['test_june_june_12_18.csv']  # ['u2.csv', 'u9.csv', 'u7.csv']
    LIVE_DATA = 'zzz_live.csv'
    HISTORICAL_DATA = '2025.csv'
    TRAIN_BEFORE_TESTING = True

    # Neural Network variables
    EPOCHS = 1 # The model trains for 20 complete passes over the dataset. More epochs allow better learning but risk overfitting
    BATCH_SIZE = 128 # The model updates weights every 32 samples instead of after each one (mini-batch gradient descent). Balances efficiency and stability.
    VALIDATION_SPLIT = 0.2 # % of the data is set aside for validation (not used in training) to check for overfitting and monitor performance.
    LAYERS = 2
    LAYERS_SIZE = 256
    DROPOUT = 0.01 # Randomly drops 20% percentage of neurons during training to prevent overfitting and improve generalization.
    WINDOW_SIZE = 100  # last prices, you're NN is looking at last x prices to make a prediction
    HORIZON = 30  # how many seconds ahead in the future you want to predict 
    LEARNING_RATE = 1e-5
    SCHEDULER = True
    WARMUP_STEPS = 0.4
    MAX_LR = 3e-4 
    MIN_LR = 1e-5
    WEIGHT_DECAY = 1e-5

    LSTM_UNITS_1 = 64
    LSTM_UNITS_2 = LSTM_UNITS_1
    KERNELS = 128  # Number of convolutional filters (kernels). Each filter detects different price movement patterns.  More kernels (filters) capture more features but increase computation.
    KERNELS_SIZE = 20 # The filter size (2-time steps) determines how many time steps to look at together.

    DIRECTION_FOCUS = 0.85
    MAGNITUDE_FOCUS = 1 - DIRECTION_FOCUS

    # Historical data variables
    FREQUENCY_DATA = '1s'  # seconds or minutes, how often you want historical data and also how far in future you want to predict 
    START_DATE = 2025, 4, 1      # 2024, 7, 21   correct format (tuple), pay attention to respect it
    END_DATE = 2025, 6, 20 


    # Portfolio variables
    WAIT_TIME = 10
    STARTING_BALANCE = 100  # Starting wallet balance in USD
    TRADE_FRACTION = 0.1  # Fraction of balance to trade
    LEVERAGE = 100
    wallet = {'USD': STARTING_BALANCE, 'BTC': 0.0}  # Initial wallet

    # Old NN
    TRAIN_EVERY_MINUTES = 59  # Retrain your NN every x minutes
    DECISION_THRESHOLD = 0.00001  # in absolute value, for smaller absolute value you're not buying nor selling