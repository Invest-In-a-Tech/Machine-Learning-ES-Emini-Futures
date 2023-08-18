
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
import joblib

##############################################
######### PROCESS HISTORICAL DATA ############
"""
This part of the code defines a class called DataFrameProcessor 
which is designed to process historical trading data
"""
class DataFrameProcessor:
    def __init__(self, file_path):
        # This attribute stores the path to the CSV file containing the historical data
        self.file_path = file_path

        # This attribute will hold the DataFrame after reading the CSV file. 
        # It is initialized as None and will be populated in the process_data method.
        self.df = None

    def process_data(self):
        # Reads the data from the CSV file into a DataFrame
        self.df = pd.read_csv(self.file_path)
        
        # Converts the 'Date' column of the DataFrame to a datetime object with a specific format 
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%Y-%m-%d %H:%M:%S')
        
        # Sets the 'Date' column as the index of the DataFrame
        # Filters the data to only include rows between the times '08:30:00' and '14:30:00'. 
        # This is to focus on a specific trading session and exclude data outside market hours.
        self.df = self.df.set_index('Date').between_time('08:30:00', '14:30:00')
        
        # Prints out the shape of the DataFrame after processing
        print(f"After processing data, DataFrame shape: {self.df.shape}")
        
        # Returns the processed DataFrame
        return self.df

    def show_data(self):
        if self.df is not None:
            print(self.df.head())
        else:
            print("No data to show")

###########################################
######### FEATURED ENGINEERING ############
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

        print(f"After feature engineering, DataFrame shape: {self.df.shape}")
        # Drop NA values
        return self.df.dropna()

##############################################
######### ENHANCED TRADING ENVIRONMENT #######
class EnhancedTradingEnv(gym.Env):

    def __init__(self, features, target, initial_balance=50000, tick_value=12.50, stop_loss=500):
        super(EnhancedTradingEnv, self).__init__()

        # Data
        self.features = features.values
        self.target = target.values
        self.feature_columns = features.columns

        # Parameters
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.tick_value = tick_value
        self.high_water_mark = initial_balance
        self.stop_loss = stop_loss

        # Action and observation spaces
        self.action_space = spaces.Discrete(5)  # Buy Entry, Buy Exit, Short Entry, Short Exit, Do Nothing
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.features.shape[1] + 2,))

        # State
        self.position = None  # 'long', 'short', or None
        self.entry_price = None
        self.current_step = None
        self.last_trade_reward = 0
        self.consecutive_losses = 0
        
        # Define reward and penalty parameters as instance attributes
        self.trade_reward_bonus = 1.5
        self.trade_reward_penalty = 0.5
        self.holding_penalty_per_step = 0.05
        self.idle_penalty = 25
        self.large_drawdown_penalty = 50
        self.consecutive_loss_penalty_multiplier = 10
        self.order_absorption_bonus_threshold = 0.05
        self.order_absorption_bonus = 1
        self.do_nothing_reward_good_decision = 0.05
        self.do_nothing_penalty_missed_trade = 0.05
        self.TIME_PENALTY_THRESHOLD = 1
        self.TIME_PENALTY = -0.1

    def reset(self):
        self.balance = self.initial_balance
        self.position = None
        self.entry_price = None
        self.current_step = 0
        self.last_trade_reward = 0
        self.consecutive_losses = 0
        self.high_water_mark = self.initial_balance  # Reset high water mark at the start of each episode
        self.entry_step = None
        return self._get_observation()

    def _get_observation(self):
        obs = np.append(self.features[self.current_step], [0, 0])
        if self.position == 'long':
            obs[-2] = 1
        elif self.position == 'short':
            obs[-1] = 1
        return obs

    def step(self, action):
        # Define reward and penalty parameters       
        reward = 0
        trade_reward = 0
        done = False
        
        #############################################
        ############# EXECUTE ACTION ################
        """
        This section of the code is responsible for defining the actions
        in a trading simulation and computing the rewards based on those actions.
        
        In essence, this code defines five possible actions in the trading environment:

            Buy Entry: Start a long position.
            Buy Exit: Close a long position.
            Short Entry: Start a short position.
            Short Exit: Close a short position.
            Do Nothing: Take no action.
            
        The rewards (profits or losses) from these trades are calculated based on the difference 
        in prices between the entry and exit actions, taking into account the direction of the trade 
        (long or short).
        """
        #############################################
        ### Execute buy action and get reward #######
        """
        If there's no open position (self.position is None), then the model decides 
        to enter a long position (buying an asset with the expectation that its price will rise).
        
        The current price (self.target[self.current_step]) is recorded as the entry price.
        """
        if self.position is None:  # No open position
            #############################################
            ### Execute buy action and get reward #####
            if action == 0:  # Buy Entry
                """
                The model decides to enter a long position (buying an asset with the expectation 
                that its price will rise, with the intention to sell it later at a higher price).
        
                The current price is recorded as the entry price for the long position.
        
                The current step is stored as the step when the buy entry was made.
                """
                self.position = 'long'
                self.entry_price = self.target[self.current_step]
                self.entry_step = self.current_step
                
            #############################################
            ### Execute short action and get reward #####
            elif action == 2:  # Short Entry
                """
                The model decides to enter a short position (selling an asset with the expectation 
                that its price will drop, with the intention to buy it back later at a lower price).
        
                The current price is recorded as the entry price for the short position.
        
                The current step is stored as the step when the short entry was made.
                """
                self.position = 'short'
                self.entry_price = self.target[self.current_step]
                self.entry_step = self.current_step
                
            ###############################################
            ### Execute do nothing action and get reward ##
            elif action == 4:  # Do Nothing
                """
                If the action is 4 (Do Nothing)
        
                If the agent chooses "Do Nothing" when it's advantageous not to enter a position:
                    If the price drops (good decision not to buy) or if the price rises (good decision not to short), the agent should be rewarded.
        
                If the agent chooses "Do Nothing" when it's disadvantageous not to enter a position:
                    If the price rises (missed opportunity to buy) or if the price drops (missed opportunity to short), the agent should be penalized.
                """
                next_price = self.target[self.current_step + 1]
                current_price = self.target[self.current_step]
        
                # Reward for not entering a bad trade
                if (next_price < current_price):  # Price dropped, good decision not to buy
                    reward += self.do_nothing_reward_good_decision
                elif (next_price > current_price):  # Price rose, good decision not to short
                    reward += self.do_nothing_reward_good_decision
                # Penalty for missing a good trade
                else:
                    reward -= self.do_nothing_penalty_missed_trade 

        elif self.position == 'long':
            ##############################################
            ### Execute buy exit action and get reward ###
            if action == 1:  # Buy Exit
                """
                If there's an open long position (self.position == 'long'), the model decides to close the position (selling the asset).
        
                The reward from this trade (trade_reward) is calculated as the difference between the current price 
                and the buy entry price, multiplied by a factor (self.tick_value). This represents the profit (or loss) 
                made from buying and then selling the asset.
        
                The position is reset to None (indicating no open position), and the entry price is also reset.
                """
                trade_reward = (self.target[self.current_step] - self.entry_price) * self.tick_value
                self.position = None
                self.entry_price = None

        elif self.position == 'short':
            ##############################################
            ### Execute short exit action and get reward #
            if action == 3:  # Short Exit
                """
                If there's an open short position (self.position == 'short'), the model decides to close the position (buying back the asset).
        
                The reward from this trade is calculated as the difference between the short entry price and the current price, multiplied by 
                a factor (self.tick_value). This represents the profit (or loss) made from short selling and then buying back the asset.
        
                The position and entry price are reset.
                """
                trade_reward = (self.entry_price - self.target[self.current_step]) * self.tick_value
                self.position = None
                self.entry_price = None 
            
        #############################################
        ############ REWARD FUNCTION ################
        """
        This reward function is designed to guide the trading model's behavior by promoting good trading habits 
        (like taking profitable trades) and discouraging bad habits (like holding positions for too long or making 
        repeated bad trades). The specific penalties and multipliers, like 1.5, 0.5 (these are arbitrary values), and the thresholds, 
        are hyperparameters that could be tuned based on how the model performs in the environment and the desired behavior.
        """
        
        #############################################
        ### Adjust rewards for profit/loss ratio ####
        """
        If the trade is profitable, I'm increasing the reward by 50% (these are arbitrary values)
        and if it's unprofitable, I'm decreasing the reward by 50%.
        
        I'm doing this to incentivize profitable trades by increasing 
        their rewards and dissuading unprofitable trades by reducing their penalties.
        """
        if trade_reward > 0:
            trade_reward *= self.trade_reward_bonus
        else:
            trade_reward *= self.trade_reward_penalty
        reward += trade_reward
        
        #############################################
        ######### Position Holding Penalty ##########
        """
        If there's an open position, a penalty is applied based on the duration 
        the position has been held. The longer a position is held, the larger the penalty. 
        
        This encourages the model to not hold positions for prolonged periods.
        """
        if self.position is not None:
            holding_duration = self.current_step - self.entry_step
            reward -= self.holding_penalty_per_step * holding_duration
            
        # Time-Based Penalty
        #if self.position is not None:
            #holding_duration = self.current_step - self.entry_step
            #if holding_duration > self.TIME_PENALTY_THRESHOLD:
                #reward += self.TIME_PENALTY
    
            
        #############################################
        ############# Idle Penalty ##################
        """
        If there's no open position and the action taken is either 1 or 3 
        (representing "do nothing" actions), a penalty of 0.05 is applied. 
        
        This encourages the model to take meaningful actions and discourages staying idle for long.
        """
        if self.position is None and (action == 1 or action == 3):
            reward -= self.idle_penalty
            
        #############################################
        ######### Large Drawdown Penalty ############
        """
        If the trade_reward is less than -100 (an arbitrary threshold for a large negative reward or drawdown), 
        a significant penalty of 50 is applied. 
        
        This discourages the model from making trades that lead to large losses.
        """
        if trade_reward < -100:  # Arbitrary threshold
            reward -= self.large_drawdown_penalty
            
        #############################################
        ##### Consecutive Losses Penalty ############
        """
        The code checks if the current trade resulted in a loss (trade_reward < 0). 
        If the last trade also resulted in a loss (self.last_trade_reward < 0), the count 
        of consecutive losses (self.consecutive_losses) is incremented. Otherwise, it is reset to 1.
        
        If the number of consecutive losses is greater than 3, a penalty of 10 is applied for each consecutive loss.
        
        This encourages the model to avoid taking trades when it's on a losing streak.
        """
        if trade_reward < 0:
            if self.last_trade_reward < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 1
        else:
            self.consecutive_losses = 0
            
        if self.consecutive_losses > 3:
            reward -= self.consecutive_loss_penalty_multiplier * self.consecutive_losses
        
        # The result of the current trade is then stored in self.last_trade_reward for use in the next step
        self.last_trade_reward = trade_reward
        
        #############################################
        ######### Order Absorption Bonus ############
        """
        If there's a significant change in the Delta or CVD, it might indicate significant market activity or order absorption. 
        By providing a bonus for profitable trades when there's significant order absorption and a penalty for unprofitable trades, 
        I'm encouraging the model to take advantage of these market conditions when they are favorable and discouraging trades when
        they are unfavorable.
        """
        # Calculate the rate of change for Delta and CVD
        delta_roc = self.features[self.current_step, list(self.feature_columns).index('ROC_Delta')]
        cvd_roc = self.features[self.current_step, list(self.feature_columns).index('ROC_CVD')]
        
        # Bonus for profitable trades, penalty for unprofitable trades
        if abs(delta_roc) > self.order_absorption_bonus_threshold or abs(cvd_roc) > self.order_absorption_bonus_threshold:
            reward += self.order_absorption_bonus if reward > 0 else -self.order_absorption_bonus 

        ###############################################
        ############# Update balance  #################
        self.balance += reward

        ###############################################
        ########### Check for stop loss ###############
        """
        This line updates the "high water mark" to the highest balance ever achieved. 
        The "high water mark" is a commonly used metric in finance to represent the peak value of an investment or trading account. 
        Every time the balance exceeds the current high water mark, it updates the high water mark to this new higher value.
        """
        # Update high water mark if the current balance exceeds it
        self.high_water_mark = max(self.high_water_mark, self.balance)

        """
        This code checks if the current balance has fallen by an amount equal to or greater 
        than the stop_loss from the high water mark. If it has, then the variable done is set to True, 
        indicating that some action (likely stopping the trading or ending the episode in a reinforcement 
        learning context) should be taken.
        """
        if self.balance <= (self.high_water_mark - self.stop_loss):
            done = True
            
        ###############################################
        ########### Check for end of data #############
        """
        This code increments the current_step and checks if it has reached or exceeded the penultimate step 
        in the target data. If so, the variable done is again set to True. This suggests that the trading strategy 
        or model operates over discrete steps or time intervals, and once all the intervals are processed, the strategy 
        ends its operation for the given dataset.
        """    
        self.current_step += 1
        if self.current_step >= len(self.target) - 1:
            done = True

        # Return the step information
        return self._get_observation(), reward, done, {}

    ###############################################
    ########### Render the environment ############
    """
    The render method is typically used in gym environments to visualize the state of the environment. In this implementation, 
    instead of providing a graphical representation, it simply prints out the current step, balance, and position to the console. 
    The default mode is set to 'human', which is common for gym environments, but I didn't use the mode argument in the function body.
    """
    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Position: {self.position}")

    ###############################################
    ########### Close the environment #############
    """
    The close method is used to perform cleanup actions when the environment is no longer needed. 
    In this implementation, the method is essentially a placeholder (with the pass statement) and doesn't do anything specific.
    I have included it here for completeness, but it's not necessary for the trading environment to function.
    """
    def close(self):
        pass

#############################################
############## TRAINING #####################
"""
This code prepares trading data for machine learning. It loads historical data, enhances it with additional features, 
and separates it into training and testing sets. The data is then normalized for consistent scale, 
and the normalization parameters are saved for future use. This preparation ensures that machine learning models can be trained effectively 
and can generalize well to new, unseen data
"""

#############################################
######### Load and process the data #########
"""
Here, the DataFrameProcessor class is instantiated with the path to a CSV file that contains historical trading data. 
The process_data method is then called on this instance. This method:

    1.) Reads the data from the specified file.
    2.) Converts the 'Date' column to a datetime object.
    3.) Filters the data to focus on a specific trading session, likely the main trading hours.
"""
processor = DataFrameProcessor("\ES.FootPrint.Renko2.df.csv")
df = processor.process_data()

################################################
##### Apply enhanced feature engineering #######
"""
This segment creates a new instance of the EnhancedFeatureEngineering class, passing the processed data df as an argument. 
The perform_feature_engineering method is then called on this instance. The method:

    1.) Generates new features based on rolling averages, standard deviations, rate of change, and MACD for specific columns.
    2.) Returns the enhanced dataframe with these new features.
"""
feature_engineer = EnhancedFeatureEngineering(df)
df_enhanced = feature_engineer.perform_feature_engineering()

#############################################
#### Define the features and the target #####
"""
The predictors or independent variables (features) are all columns of the enhanced dataframe df_enhanced except 
for the 'Close' column. The 'Close' column, typically representing closing prices in trading, is taken as 
the target variable or the dependent variable we want to predict or work with.
"""
features = df_enhanced.drop('Close', axis=1)
target = df_enhanced['Close']

#############################################
######## Setting a random seed ##############
"""
This line sets a fixed random seed for operations that have a random component, ensuring that they produce the same 
results every time they're run.

Reproducibility: By getting the same split every time, we can ensure that any changes in model performance are due to 
modifications in the model or preprocessing steps, and not because of a different random split of the data.
"""
seed = 42

###################################################
## Split the data into training and testing sets ##
"""
The dataset is split into training and testing subsets. Notably, the data isn't shuffled (shuffle=False), 
which is typical for time-series data where order matters. A fixed random state ensures reproducibility.
"""
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.2, random_state=seed, shuffle=False
)

###################################################
##### Scale the features to the range [0, 1] ######
"""
The features are scaled to lie within the range [0, 1] using the MinMaxScaler. This is essential for many machine 
learning models to ensure features have the same scale. The scaler is fit only on the training data to prevent data leakage.
"""
scaler = MinMaxScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

###################################################
### Create dataframes for the scaled features #####
"""
The scaled features, which are in the form of NumPy arrays, are converted back into pandas DataFrames. 
This step is necessary for retaining the original column names and indices.
"""
features_train_scaled = pd.DataFrame(features_train_scaled, columns=features_train.columns, index=features_train.index)
features_test_scaled = pd.DataFrame(features_test_scaled, columns=features_test.columns, index=features_test.index)
print(f"After scaling, training data shape: {features_train_scaled.shape}, test data shape: {features_test_scaled.shape}")

###################################################
########## Save the scaler object #################
"""
The fitted MinMaxScaler is saved to disk using joblib.dump(). This is crucial because, during inference or real-time trading, 
any new data must be scaled using the same scaler that was fit on the training data.
"""
scaler_filename = ""
joblib.dump(scaler, scaler_filename)

#############################################
######### EnhancedTradingEnv ################
"""
This section of the code is focused on setting up and training the reinforcement learning agent using the PPO algorithm for a trading task. 
The agent learns to make trading decisions based on historical data, with the goal of maximizing cumulative reward, which is profit in this context. 
Once trained, the model will be used to make future trading decisions or further refined with more data.
"""

#############################################
## Create the enhanced trading environment ##
"""
Here, an instance of the EnhancedTradingEnv class is created, which represents a custom trading environment. 
This environment is compatible with the OpenAI Gym framework, a popular library for developing and comparing reinforcement learning (RL) algorithms.

    1.) features_train_scaled: This represents the processed and scaled feature data that will be used to make decisions in the environment.
    2.) target_train: This represents the target data, which in this trading environment, is the 'Close' price.
        This environment will take actions, observe the results, and receive rewards based on the historical data provided, 
        simulating the process of trading over the training dataset.
"""
env = EnhancedTradingEnv(features_train_scaled, target_train)

#############################################
######## Initializing (PPO) Model ###########
"""
Here, the model is being instantiated using the Proximal Policy Optimization (PPO) algorithm, which is a popular and 
effective RL algorithm for training agents in various environments.

    1.) "MlpPolicy": This specifies the type of neural network policy to be used with the PPO algorithm. In this case, 
        it's a multi-layer perceptron (MLP) policy.
    2.) env: This is the environment in which the agent will be trained, which is our EnhancedTradingEnv.
    3.) verbose=1: This setting ensures that the training process will print out logs so you can monitor the training.

There are also hyperparameters that I commented-out like learning_rate, n_steps, n_epochs, etc. These can be uncommented 
and adjusted to fine-tune the PPO algorithm's behavior.
"""
model = PPO("MlpPolicy", env, verbose=1, 
            #learning_rate=0.0006, # default: 0.0003
            #n_steps=2048, # default: 2048
            #n_epochs=10, # default: 10
            #gamma=0.95, # default: 0.99
            #gae_lambda=0.95, # default: 0.95
            #clip_range=0.2, # default: 0.2
            #ent_coef=0.01 # default: 0.0
            ) 

###################################################
############# Training the model ##################
"""
The learn method trains the PPO model. The agent will interact with the environment, take actions, receive rewards, 
and adjust its policy for a total of 100,000 timesteps. Each timestep corresponds to an interaction with the environment.
"""
model.learn(total_timesteps=100000)

#############################################
####### Saving the Trained Model ############
"""
After training, the model is saved to disk using the specified path. This allows for later use without having to retrain the model. 
The saved model will contain the learned policy, and it can be loaded back to make predictions, continue training, or evaluate its 
performance on new data.
"""
model.save()

##################################################
#### Save reward/penalty parameters to a file ####
file_path2 = ""
with open(file_path2, "w") as param_file:
    param_file.write("Reward and Penalty Parameters:\n\n")
    param_file.write(f"Trade Reward Bonus (Profitable): {env.trade_reward_bonus}\n")
    param_file.write(f"Trade Reward Penalty (Unprofitable): {env.trade_reward_penalty}\n")
    param_file.write(f"Position Holding Penalty (per step): {env.holding_penalty_per_step}\n")
    param_file.write(f"Idle Penalty (for invalid exit action): {env.idle_penalty}\n")
    param_file.write(f"Large Drawdown Penalty (when trade reward < -100): {env.large_drawdown_penalty}\n")
    param_file.write(f"Consecutive Loss Penalty (when more than 2 consecutive losses): {env.consecutive_loss_penalty_multiplier} * number of consecutive losses\n")
    param_file.write(f"Order Absorption Bonus (when delta_roc or cvd_roc > {env.order_absorption_bonus_threshold}): ±{env.order_absorption_bonus} depending on the trade profitability\n")
    param_file.write(f"Do Nothing Reward (Good Decision): {env.do_nothing_reward_good_decision}\n")
    param_file.write(f"Do Nothing Penalty (Missed Good Trade): {env.do_nothing_penalty_missed_trade}\n")
    

#############################################
############ Test the model #################
"""
This code section is simulating the trained model's interactions with the trading environment for 1000 steps. 
It's a way to evaluate the model's performance in a controlled setting post-training. By observing the actions taken 
and the rewards received, one can gauge how well the model might perform in a real-world trading scenario.
"""

#############################################
### Resetting the environment for testing ####
"""
The reset() method initializes the environment to its starting state and returns the initial observation. 
This observation is crucial as it provides the initial state information to the model. The shape of this 
observation is printed out, which can give insights into the number of features or data points the model considers at each step.
"""
obs = env.reset()
print(f"Testing model, initial observation shape: {obs.shape}")

##############################################
###### Initialize Testing Variables ##########
"""
1.) total_rewards: Keeps track of the cumulative reward during the testing phase.
2.) i: A counter to ensure the testing loop runs for a specific number of steps (in this case, 1000).
"""
total_rewards = 0
i = 0

############################################
############## Testing Loop ################
"""

"""
while i < 1000:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    total_rewards += reward
    print(f'Step: {i}, Action: {action}, Reward: {reward}, Total Rewards: {total_rewards}, Balance: {env.balance}')

    if done:
        print('Episode ended')
        obs = env.reset()  # Resetting the environment
        total_rewards = 0  # Reset total rewards for the new episode
    else:
        i += 1
