import pandas as pd
from trade29.sc.bridge import SCBridge
import time
import numpy as np
from datetime import datetime
from enhanced_feature_engineering import EnhancedFeatureEngineering
from stable_baselines3 import PPO
from sklearn.preprocessing import MinMaxScaler
import joblib

############################################
######## write_to_file function ############
def write_to_file(msg, filename='output.txt'):
    with open(filename, 'a') as f:
        f.write(str(msg) + '\n')

############################################
############# LIVE TRADING #################
# Load the pre-trained model and scaler
model_path = ""
scaler_filename = ""
loaded_model = PPO.load(model_path)
scaler = joblib.load(scaler_filename)

class TradeDataProcessor:
    def __init__(self, start_time="08:30:00", end_time="14:30:00"):
        # Initialize instance variables
        self.df = None
        self.start_time = datetime.strptime(start_time, "%H:%M:%S").time()
        self.end_time = datetime.strptime(end_time, "%H:%M:%S").time()
        self.bridge = SCBridge()
        self.current_position = 0
        self.model = loaded_model
        self.scaler = scaler
        
        # Request data from SCBridge
        self.data_id = self.bridge.graph_data_request(
            key='key', base_data='1;2;3;4;5', sg_data="ID2.[SG1;SG10]", 
            historical_init_bars=50, realtime_update_bars=50, on_bar_close=True, 
            update_frequency=0, include_bar_index=True
        )
        self.response_q = self.bridge.get_response_queue()
    
    #########################################
    ########## PROCESS LIVE DATA ############   
    def process_data(self):
        while True:
            # Get data from SCBridge
            msg = self.response_q.get()
            
            # Check if the data is for the requested data_id
            if msg.request_id != self.data_id:
                print(msg)
                write_to_file(msg)
                continue
            
            # Process the raw data
            raw_df = msg.df
            self.df = raw_df.rename(columns={0: 'Date', 3: 'Open', 4: 'High', 5: 'Low', 6: 'Close', 
                                        7: 'Volume', 8: 'Delta', 9: 'CVD'})
            self.df = self.df.sort_values('Date')    
            self.df['Date'] = pd.to_datetime(self.df['Date'], format='%Y-%m-%d %H:%M:%S')
            self.df.drop(columns=[1, 2], inplace=True)
            self.df.set_index('Date', inplace=True)
            
            # Print the processed data
            if self.df is not None:
                print(f"After processing data, DataFrame shape: {self.df.shape}")
                write_to_file(f"After processing data, DataFrame shape: {self.df.shape}")
            else:
                print("No data to show")
                write_to_file("No data to show")
                
            #########################################
            ####### FEATURED ENGINEERING ############            
            # Create an instance of the EnhancedFeatureEngineering class with the self.df dataframe as input    
            feature_engineer = EnhancedFeatureEngineering(self.df) 
            df_enhanced = feature_engineer.perform_feature_engineering()
            
            # Print the shape of the enhanced dataframe
            print(f"After feature engineering, DataFrame shape: {df_enhanced.shape}")
            write_to_file(f"After feature engineering, DataFrame shape: {df_enhanced.shape}")
            
            #########################################
            ############# PREDICTION ################
            # Extract and scale the features
            live_features = df_enhanced.drop('Close', axis=1)
            live_features_scaled = self.scaler.transform(live_features)
            
            # Print the shape of the features and the scaled features
            print(f"Shape of live features: {live_features.shape}")
            print(f"Shape of scaled live features: {live_features_scaled.shape}")
            write_to_file(f"Shape of live features: {live_features.shape}")
            write_to_file(f"Shape of scaled live features: {live_features_scaled.shape}")
            
            # Combine the scaled features with position and account_balance for prediction
            obs = np.hstack((live_features_scaled[0], [self.current_position, 50000]))  # adjust the 50000 if needed
            
            # Print the shape of the observation before prediction
            print(f"Shape of observation for prediction: {obs.shape}")
            write_to_file(f"Shape of observation for prediction: {obs.shape}")           

            # Predict actions with the agent
            action, _states = self.model.predict(obs)
            
            # Print the action predicted by the model
            print(f"Predicted action: {action}") 
            write_to_file(f"Predicted action: {action}")           

            #########################################
            ########### EXECUTE ACTIONS #############
            """
            
            """
            #############################################
            ########### Check Trading Hours #############
            """
            In essence, this code ensures that any subsequent operations are only 
            carried out during the specified trading hours. It's a common practice in algorithmic 
            trading to restrict trading actions to certain hours, especially when dealing with markets that have 
            specific opening and closing times.
            """
            # Get the time from latest data row
            current_time = self.df.index[-1].time()
            """
            This line gets the time from the last (most recent) row of the dataframe self.df. 
            The assumption here is that the dataframe's index is of datetime type, and the last row 
            corresponds to the latest timestamp.
            """
            
            # Check if the current time is within the trading hours
            if self.start_time <= current_time <= self.end_time:
                """
                This checks if the extracted current_time is within specified trading hours (self.start_time to self.end_time).
                Any actions or operations following this condition will only be executed if the current time falls within these 
                trading hours.
                """

                #########################################
                ########## No Open Position #############
                if self.current_position == 0:
                    """
                    Buy Entry (action == 0): A buy order is submitted, 
                    and the position is set to 1 (indicating a long position).
                    """
                    if action == 0:  # Buy entry
                        print("Long condition")
                        write_to_file("Long condition")
                        self.bridge.submit_order('es.key', is_buy=True, qty=1)
                        self.current_position = 1
                        """
                    Short Entry (action == 2): A sell (short) order is submitted, 
                    and the position is set to -1 (indicating a short position).    
                        """
                    elif action == 2:  # Short entry
                        print("Short condition")
                        write_to_file("Short condition")
                        self.bridge.submit_order('es.key', is_buy=False, qty=1)
                        self.current_position = -1
                        """
                    No Action (action == 4): No action is taken, and a message 
                    indicating this is printed.   
                        """
                    elif action == 4:
                        print("No action")
                        write_to_file("No action")
                        
                #########################################
                ######### Long Position Open ############
                elif self.current_position == 1:  # Long position is open
                    if action == 1:  # Buy exit
                        self.bridge.flatten_and_cancel('es.key')  # Cancel all open orders
                        print("Long exit")
                        write_to_file("Long exit")
                        self.current_position = 0
                        
                #########################################
                ######### Short Position Open ###########
                elif self.current_position == -1:  # Short position is open
                    if action == 3:  # Short exit
                        self.bridge.flatten_and_cancel('es.key')  # Cancel all open orders
                        print("Short exit")
                        write_to_file("Short exit")
                        self.current_position = 0  
                        
            print("=====================================================")
            print(df_enhanced.tail(1))
            write_to_file("=====================================================")
            write_to_file(df_enhanced.tail(1))
       
# Create a new instance of the TradeDataProcessor class
processor = TradeDataProcessor()

# Start processing data
processor.process_data()
