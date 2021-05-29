# Crypto Portfolio Backtesting

This is my initial attempt at adapting my portfolio construction strategies from the world of stocks into the world of crypto.

Run **backtest_main_program_crypto.py** to see it in action.

OR take a look at the jupyter notebook I've now added to see a high level view of results.

If you look at my JSE share portfolio repository, you can read about the strategy used there, both the QM and the QV variants. Obviously in the world of crypto we don't have financial data to draw on, like we did with the stocks.
However, the QM strategy can be run with the exact same coding, except that one might want to shift the time frame from monthly down to daily or even less.

For the QV strategy, I'm curious about exploring different ways of implementing this - perhaps by using TVL vs coin price, or some similar analog of value in the crypto space.

I'm also curious to see how the different weightings affect portfolio construction in crypto. At the moment I also have a crypto portfolio running on shrimpy.io, which is a basic threshold rebalancing system. I'd love to create something a bit more nuanced here.

For this implementation, I looked to keep as much of the logic the same as before - using ND arrays, starting with the whole universe's prices against time. Again, I've shifted from monthly to daily prices here, because of the much higher volatility in the crypto space.
Prices are imported via the coingecko API. I have done some readings on alternative sources of prices, and will look to experiment with these soon - potentially Binance, as I might be able to implement trades through them as well.
That being said, I'd love to see if I can avoid CEX's altogether and use Dex's (preferably on polygon to keep gas fees low). This would allow me to keep this whole system running in a non-custodial fashion.

Still to do:
- so much!
- come up with a value metric for crypto (perhaps will vary per crypto 'sector')
- enlarge the universe of coins being imported
- filter that universe by market cap and liquidity, to choose just the largest, most liquid 100-500 coins
- filter out coins that are actually indices, leverage plays etc.
- go back further in history
- try out different API data providers
- investigate turning this into a portfolio contruction tool (not just backtesting tool), i.e. have it trade in and out of the market on my behalf, automatically
