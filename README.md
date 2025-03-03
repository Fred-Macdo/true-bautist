# True Bautist - A [Lumibot](https://github.com/Lumiwealth/lumibot) Adaptation

Being a fucking donkey is hard but fortunately for you I am here to help. I built this buggy little application to help you algo-trade your way into insolvency, destitution and divorce or by some infinitesimal chance, wealth accumulation. I was sober for much of the development period but none of the testing (I didn't test) so expect some shit to go awry and deal with it. Leveraging a great open-source framework called Lumibot as a back-end engine, I hacked together a configuration file-based system that allows noob algo-traders to develop their own strategies. "Traders" can identify *stocks* to trade,  select own *indicators* to track, specify *entry/exit conditions* for their positions and set *risk-management* guardrails for their trades.

### How do I use the True Bautist?

I did most of the work for you already so all you need to do is follow a few steps to get this thing working.

1. Clone the repo:

`git clone https://github.com/Fred-Macdo/true-bautist.git`

2. Edit your config file:

- Start with the trading universe or overall configurations.

```
# TRADING UNIVERSE
symbols: ["PLTR", "SOFI", "HIMS"]
timeframe: "1D" #keys = ["1Min", "5Min", "15Min", "30Min", "1h", "1d", "2d", "1w", "1month"]
start_date: "2024-01-01"
end_date: "2025-01-01"
```

- Choose the indicators you want to include and their respective parameters

```
# REQUIRED INDICATORS
indicators:
  - name: "EMA"
    params:
      period: 5
  - name: "SMA"
    params:
      period: 20
  - name: "RSI"
    params:
      period: 14
  - name: "BBANDS"
    params:
      period: 20
      std_dev: 2
  - name: "ATR"
    params:
      period: 14
```

- Next, specify the conditions which you want your algo to enter a trade and the conditions which trigger an exit.

```
# ENTRY CONDITIONS
entry_conditions:
  - indicator: "close"
    comparison: "above"
    value: "sma_20"
  - indicator: "ema_5"
    comparison: "above"
    value: sma_20

# EXIT CONDITIONS
exit_conditions:
  - indicator: "close"
    comparison: "below"
    value: "sma_20"
  - indicator: "ema_5"
    comparison: "crosses_below"
    value: "sma_20"
  # Price breaking below bands
  - indicator: "close"
    comparison: "below"
    value: "lowerband"
```

- Finally, tweak the risk-management framework which will help you protect your assets during trading. **NOTE:** `risk_per_trade`, `stop_loss`, and `take_profit` are expressed as percents.

3. Run the backtest using the --backtest flag. 

`python true_bautist.py -c <configuration.yaml> -k <api_keys.yaml>`

**NOTE:** Use the `--live` flag to run your strategy live. In addition you will need to add your `<api_keys.yaml>` after the `configuration.yaml` file while running on the command line. I have set it up for [Alpaca Trading](https://app.alpaca.markets/signup). Alpaca is an API First Brokerage that enables completely free algo trading.
