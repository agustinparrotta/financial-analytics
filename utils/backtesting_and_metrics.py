import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
import quantstats as qs
from .sharpe_ratio_stats import *



def daily_return(prices):
    df0 = prices.index.searchsorted(prices.index-pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(prices.index[df0-1],
                    index=prices.index[prices.shape[0]-df0.shape[0]:])
    df0 = prices.loc[df0.index] / prices.loc[df0.values].values-1  # Daily returns
    return df0

def run_sample_and_hold_strategy(prices, budget):
    df0 = prices.copy(deep=True)
    df0.rename('Value')
    num_assets = budget / df0[0]
    df0 *= num_assets
    return df0


def strategy_report(st_returns, st_name, underlying_asset_returns=pd.Series(dtype='float64')):
    print('-------------------------------------------------------------------')
    print('-------------------------------------------------------------------')
    print('Strategy: {}'.format(st_name))
    print('Sharpe: {}'.format(qs.stats.sharpe(st_returns, periods=365, annualize=True)))
    print('Sortino: {}'.format(qs.stats.sortino(st_returns)))
    print('Adjusted Sortino: {}'.format(qs.stats.adjusted_sortino(st_returns)))
    print('Win loss ratio: {}'.format(qs.stats.win_loss_ratio(st_returns)))
    print('Win rate: {}'.format(qs.stats.win_rate(st_returns)))
    print('Avg loss: {}'.format(qs.stats.avg_loss(st_returns)))
    print('Avg win: {}'.format(qs.stats.avg_win(st_returns)))
    print('Avg return: {}'.format(qs.stats.avg_return(st_returns)))
    print('Volatility: {}'.format(qs.stats.volatility(st_returns, periods=st_returns.shape[0], annualize=True)))
    print('Value at risk: {}'.format(qs.stats.value_at_risk(st_returns, sigma=1, confidence=0.95)))
    if not underlying_asset_returns.empty:
        df = pd.merge(st_returns, underlying_asset_returns, how='inner', left_index=True, right_index=True)
        print('Correlation to underlying: {}'.format(df.corr()))
    print('-------------------------------------------------------------------')
    print('Sharpe: {}'.format(estimated_sharpe_ratio(st_returns)))
    print('Annualized Sharpe: {}'.format(ann_estimated_sharpe_ratio(st_returns, periods=365)))
    print('STDDEV Sharpe: {}'.format(estimated_sharpe_ratio_stdev(returns=st_returns)))
    psrs = [probabilistic_sharpe_ratio(returns=st_returns, sr_benchmark=float(i)/100.) for i in range(0, 101)]
    print('PSR: {}'.format(psrs))
    print('-------------------------------------------------------------------')
    print('Mean return: {}'.format(qs.stats.avg_return(st_returns)))
    print('Variance of returns: {}'.format(qs.stats.volatility(st_returns, annualize=False) ** 2))
    print('Skewness of returns: {}'.format(scipy_stats.skew(st_returns, nan_policy='omit')))
    print('Kurtosis of returns: {}'.format(scipy_stats.kurtosis(st_returns, nan_policy='omit')))
    print('-------------------------------------------------------------------')

def plot_returns(returns, strategy_name):
    plt.hist(returns, 100, facecolor='blue', alpha=0.7, log=True)
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.title('Histogram of returns for {}'.format(strategy_name))
    plt.grid(True)
    plt.show()
    
def plot_psr(base_returns, st_returns):
    psr_base = np.asarray([probabilistic_sharpe_ratio(returns=base_returns, sr_benchmark=float(i)/100.) for i in range(0, 101)])
    psr_st = np.asarray([probabilistic_sharpe_ratio(returns=st_returns, sr_benchmark=float(i)/100.) for i in range(0, 101)])

    psr_base_odds = np.divide(psr_base, 1. - psr_base)
    psr_st_odds = np.divide(psr_st, 1. - psr_st)
    psr_base_odds_log = np.log10(psr_base_odds)
    psr_st_odds_log = np.log10(psr_st_odds)

    x = np.asarray([i / 100. for i in range(0, 101)])

    plt.plot(x, psr_base_odds_log, color='blue', label='Odds ratio PSR buy and hold')
    plt.plot(x, psr_st_odds_log, color='red', label='Odds ratio PSR strategy under test')
    plt.legend()
    plt.xlabel('Target Sharpe Ratio')
    plt.ylabel('Odds ratio of Probabilistic Sharpe Ratio')
    plt.title('Odds ratio of Probabilistic Sharpe Ratio')
    plt.grid()
    plt.show()

def plot_value(df):
    df.plot(y='Value', color='blue')
    plt.xlabel('Timestampt')
    plt.ylabel('Portfolio value [USD]')
    plt.title('Portfolio valuation')
    plt.grid()
    plt.show()

def run_self_funded_strategy(df, params):
    i=0
    last_portfolio_value = 1
    tx_costs = 0.
    
    for index,row in df.iterrows():
        if row['bets_usd'] > 0:
            bet = row['bets_usd']
            initial_price = row['Close']
            price_i = row['Close']
            # Will track the date index
            j = 1
            
            # Pays to enter the position
            df.loc[df.index[i], 'Value'] = df.loc[df.index[i], 'Value'] - row['bets_usd'] * params['buy_fee']
            tx_costs = tx_costs + np.abs(row['bets_usd'] * params['buy_fee'])

            # After t1_length without touching any of the barrier, the position
            # is dismantled. If it if happens before t1_length, then we do it earlier.
            while price_i < (initial_price * (1 + params['pt'])) and price_i > (initial_price * (1 - params['sl'])) and (j < params['t1_length'] + 1): #Cuando se cumple la condición, salimos de la posición
                price_i = df.loc[df.index[i+j], 'Close']
                df.loc[df.index[i+j], 'Value'] = df.loc[df.index[i], 'Value'] + row['bets_usd'] * (price_i / initial_price - 1.)
                last_portfolio_value = df.loc[df.index[i+j], 'Value']
                j = j + 1       
            
            # Fee to leave the position.
            df.loc[df.index[i+j-1]:, 'Value'] = last_portfolio_value - row['bets_usd'] * (price_i / initial_price) * params['sell_fee']
            tx_costs = tx_costs + np.abs(row['bets_usd'] * (price_i / initial_price) * params['sell_fee'])
        
        if row['bets_usd'] < 0:
            bet = row['bets_usd']
            initial_price = row['Close']
            price_i = row['Close']
            j = 1 # Will track the date index
            
            # Shorting BTC. Two fees needs to be paids, the sell + the loan fee.
            df.loc[df.index[i], 'Value'] = df.loc[df.index[i], 'Value'] - np.abs(row['bets_usd']) * params['sell_fee'] - np.abs(row['bets_usd']) * params['short_fee']
            tx_costs = tx_costs + np.abs(row['bets_usd']) * params['sell_fee'] + np.abs(row['bets_usd']) * params['short_fee']

            # Same as before, waiting for any barrier to be touched.
            while price_i < (initial_price * (1 + params['pt'])) and price_i > (initial_price * (1 - params['sl'])) and (j < params['t1_length'] + 1): #Cuando se cumple la condición, salimos de la posición
                price_i = df.loc[df.index[i+j], 'Close']
                df.loc[df.index[i+j], 'Value'] = df.loc[df.index[i], 'Value'] + row['bets_usd'] * (price_i / initial_price - 1.)
                last_portfolio_value = df.loc[df.index[i+j], 'Value']
                j = j + 1   
            # We pay the commission to leave the position.
            df.loc[df.index[i+j-1]:, 'Value'] = last_portfolio_value - np.abs(row['bets_usd']) * (price_i / initial_price) * params['buy_fee']
            tx_costs = tx_costs + np.abs(row['bets_usd']) * (price_i / initial_price) * params['buy_fee']

        # If there is no money, we stop.
        if last_portfolio_value <= 0.:
            break
        i = i + 1 # Moves forward with the next row.

    return df, tx_costs

def build_portfolio_self_funded_df(prices, bets, params):
    # Refactor the bets so they can be worked out.
    bets_df = bets.to_frame()
    bets_df.index.name = 'Timestamp'
    bets_df.rename(columns={0:'bets'}, inplace=True)

    # Generate a new df
    df = prices.to_frame().copy()
    df = pd.merge(df, bets_df, how='left', left_index=True, right_index=True)
    df['bets_usd'] = df['bets'] * params['budget']
    df['Value'] = params['budget']
    df.fillna(0, inplace=True)

    df, tx_costs = run_self_funded_strategy(df, params)
    df['rets'] = daily_return(df['Value'])

    return df, tx_costs
    
