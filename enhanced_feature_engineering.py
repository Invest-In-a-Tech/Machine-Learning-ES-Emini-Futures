import numpy as np

class EnhancedFeatureEngineering:
    def __init__(self, df):
        self.df = df.copy()

    def perform_feature_engineering(self):
        # Original Features
        self.df['RollingMeanVolume'] = self.df['Volume'].rolling(window=20).mean()
        self.df['RollingStdVolume'] = self.df['Volume'].rolling(window=20).std()
        self.df['RollingMeanDelta'] = self.df['Delta'].rolling(window=20).mean()
        self.df['RollingStdDelta'] = self.df['Delta'].rolling(window=20).std()
        self.df['RollingMeanCVD'] = self.df['CVD'].rolling(window=20).mean()
        self.df['RollingStdCVD'] = self.df['CVD'].rolling(window=20).std()

        # Rate of Change
        for column in ['Volume', 'Delta', 'CVD']:
            self.df[f'ROC_{column}'] = self.df[column].pct_change()
            # Handle infinite values resulting from division by zero
            self.df[f'ROC_{column}'].replace([np.inf, -np.inf], np.nan, inplace=True)
            # For this example, we're filling NaN values with 0; however, you can choose another method if preferred
            self.df[f'ROC_{column}'].fillna(0, inplace=True)

        # Moving Averages
        for column in ['Delta', 'CVD']:
            self.df[f'MA50_{column}'] = self.df[column].rolling(window=50).mean()

        # MACD for Delta and CVD
        for column in ['Delta', 'CVD']:
            self.df[f'MACD_{column}'] = self.df[column].ewm(span=12, adjust=False).mean() - self.df[column].ewm(span=26, adjust=False).mean()
            self.df[f'Signal_{column}'] = self.df[f'MACD_{column}'].ewm(span=9, adjust=False).mean()

        # Drop NA values
        return self.df.dropna()