import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from environments.forex_env import ForexEnv
from agents.ppo_agent import PPOAgent
from agents.rule_based import RuleBasedAgent
from data.preprocessor import load_forex_data
from utils.config import load_config
from models.model_manager import ModelManager

def create_plot_folder():
    folder = "backtest_plots"
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def plot_price_with_trades(test_data, trade_log):
    plt.figure(figsize=(14, 7))
    # Plot close prices
    plt.plot(test_data.index, test_data['close'], label='Close Price', color='blue', alpha=0.7)
    
    # Plot Long and Short trades
    for trade in trade_log:
        if trade['trade_type'] == 'long':
            plt.scatter(trade['open_time'], trade['open_price'], color='green', marker='^', s=100, label='Long Open (Buy)' if 'Long Open (Buy)' not in plt.gca().get_legend_handles_labels()[1] else '')
            plt.scatter(trade['close_time'], trade['close_price'], color='lime', marker='v', s=100, label='Long Close' if 'Long Close' not in plt.gca().get_legend_handles_labels()[1] else '')
        elif trade['trade_type'] == 'short':
            plt.scatter(trade['open_time'], trade['open_price'], color='red', marker='v', s=100, label='Short Open (Sell)' if 'Short Open (Sell)' not in plt.gca().get_legend_handles_labels()[1] else '')
            plt.scatter(trade['close_time'], trade['close_price'], color='orange', marker='^', s=100, label='Short Close' if 'Short Close' not in plt.gca().get_legend_handles_labels()[1] else '')
    
    plt.title('EUR/USD Price with Long/Short Trades (2025)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    folder = create_plot_folder()
    plt.savefig(f"{folder}/price_trades_2025.png", dpi=300)
    
    # Show plot
    plt.show()
    plt.close()

def run_backtest(pair="EURUSD", model_name="ppo_EURUSD_2025", use_rule_based=False):
    config = load_config()
    data = load_forex_data()
    
    test_data = data[data.index >= '2025-01-01']
    print(f"Testing data: {len(test_data)} rows")
    
    env = ForexEnv(
        data=test_data,
        initial_balance=config["initial_balance"],
        daily_loss_limit=config["challenge"]["daily_loss_limit"],
        max_drawdown=config["challenge"]["max_drawdown"]
    )
    
    if use_rule_based:
        agent = RuleBasedAgent(env)
    else:
        model_manager = ModelManager()
        agent = PPOAgent(env)
        agent.load(f"models/saved_models/{model_name}/{model_name}.zip")
    
    state, _ = env.reset()
    done = False
    trades = 0
    daily_pnl = []
    trading_days = set()
    current_day = None
    trade_profits = []
    action_counts = {0: 0, 1: 0, 2: 0}
    trade_log = []
    position_open = None
    
    while not done:
        action = agent.predict(state, deterministic=False)
        action_counts[action] += 1
        prev_balance = env.balance
        current_price = test_data.iloc[env.current_step]['close']
        current_time = test_data.index[env.current_step]
        
        if action == 1 and env.position == 0 and position_open is None:
            position_open = {
                'action': action,
                'trade_type': 'long',
                'open_time': current_time,
                'open_price': current_price + (env.spread * 0.0001)
            }
        elif action == 2 and env.position == 0 and position_open is None:
            position_open = {
                'action': action,
                'trade_type': 'short',
                'open_time': current_time,
                'open_price': current_price - (env.spread * 0.0001)
            }
        
        state, reward, done, truncated, info = env.step(action)
        
        action_str = {0: "Hold", 1: "Buy", 2: "Sell"}[action]
        day = test_data.index[env.current_step].date()
        
        if action in [1, 2] and env.position == 0 and prev_balance != info["balance"] and position_open:
            trade_profit = info["balance"] - prev_balance
            trades += 1
            trading_days.add(day)
            trade_profits.append(trade_profit)
            trade_log.append({
                'trade_number': trades,
                'trade_type': position_open['trade_type'],
                'open_time': position_open['open_time'],
                'open_price': position_open['open_price'],
                'close_time': current_time,
                'close_price': current_price,
                'spread': env.spread * 10,  # $10 for 1 pip
                'profit': trade_profit,
                'balance': info["balance"]
            })
            print(f"Trade {trades}, Type: {position_open['trade_type'].capitalize()}, Action: {action_str}, Profit: {trade_profit:.2f}, Open: {position_open['open_price']:.5f}, Close: {current_price:.5f}")
            position_open = None
        
        if current_day != day:
            if current_day is not None:
                daily_pnl.append(env.daily_pnl)
            env.daily_pnl = 0
            current_day = day
        
        profit_pct = (info["balance"] - config["initial_balance"]) / config["initial_balance"] * 100
        if profit_pct >= config["challenge"]["profit_target"] * 100 and len(trading_days) >= config["challenge"]["min_trading_days"]:
            done = True
            print(f"Challenge passed at step {env.current_step}: Profit {profit_pct:.2f}%, Days {len(trading_days)}")
        
        if env.current_step % 24 == 0 or done:
            print(f"Step {env.current_step}, Day {len(daily_pnl) + 1}, Action: {action_str}, Balance: {info['balance']:.2f}, Equity: {info['equity']:.2f}, Trades: {trades}")
    
    if current_day is not None:
        daily_pnl.append(env.daily_pnl)
    
    profit_pct = (info["balance"] - config["initial_balance"]) / config["initial_balance"] * 100
    max_daily_loss = min(daily_pnl) / config["initial_balance"] * 100 if daily_pnl else 0
    avg_trade_profit = sum(trade_profits) / len(trade_profits) if trade_profits else 0
    total_steps = sum(action_counts.values())
    
    print(f"\nBacktest Results:")
    print(f"Profit: {profit_pct:.2f}%")
    print(f"Max Daily Loss: {max_daily_loss:.2f}%")
    print(f"Trading Days: {len(trading_days)}")
    print(f"Total Trades: {trades}")
    print(f"Avg Trade Profit: {avg_trade_profit:.2f}")
    print(f"Action Counts: {action_counts} (Total: {total_steps})")
    print(f"Success: {profit_pct >= config['challenge']['profit_target'] * 100 and max_daily_loss > -config['challenge']['daily_loss_limit'] * 100 and len(trading_days) >= config['challenge']['min_trading_days']}")
    
    # Trade Analysis Table
    print("\nTrade Analysis:")
    print("-" * 90)
    print(f"{'Trade':<6} {'Type':<6} {'Open Time':<20} {'Open Price':<11} {'Close Time':<20} {'Close Price':<12} {'Spread':<7} {'Profit':<7} {'Balance':<9}")
    print("-" * 90)
    for trade in trade_log:
        print(f"{trade['trade_number']:<6} {trade['trade_type'].capitalize():<6} {trade['open_time'].strftime('%Y-%m-%d %H:%M'):<20} {trade['open_price']:<11.5f} {trade['close_time'].strftime('%Y-%m-%d %H:%M'):<20} {trade['close_price']:<12.5f} {trade['spread']:<7.2f} {trade['profit']:<7.2f} {trade['balance']:<9.2f}")
    print("-" * 90)
    
    # Save Trade Analysis to Excel
    folder = create_plot_folder()
    trade_df = pd.DataFrame(trade_log)
    trade_df['open_time'] = trade_df['open_time'].dt.strftime('%Y-%m-%d %H:%M')
    trade_df['close_time'] = trade_df['close_time'].dt.strftime('%Y-%m-%d %H:%M')
    trade_df = trade_df[['trade_number', 'trade_type', 'open_time', 'open_price', 'close_time', 'close_price', 'spread', 'profit', 'balance']]
    trade_df.to_excel(f"{folder}/trade_analysis.xlsx", index=False)
    print(f"Trade analysis saved to {folder}/trade_analysis.xlsx")
    
    # Generate price plot
    plot_price_with_trades(test_data, trade_log)
    
    return profit_pct, len(trading_days), trades, avg_trade_profit, action_counts

if __name__ == "__main__":
    run_backtest()