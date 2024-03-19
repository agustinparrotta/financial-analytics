# Financial Analytics

A pairs trading strategy using Bitcoin and its proxies is addressed. These strategies aim to exploit short-term differences between highly correlated assets, taking a _long_ position on the underperforming asset and a _short_ position on the overperforming one. Different proxies of Bitcoin are analyzed:

- Cryptocurrencies: Ethereum (ETH)
- Microstrategy (MSTR)
- Grayscale Bitcoin Investment Trust (GBTC)
- Bitcoin mining companies: MARA, RIOT
- ETFs: BITO, BITW

However, GBTC is ultimately used as the sole proxy due to its high correlation.

### Resources:

- Data source: Coin Metrics for cryptocurrency price series and Yahoo Finance for the rest.
- Source code: Relied on the codes provided in the book _Advances in Financial Machine Learning_ by Marcos López de Prado.
- Pairs trading algorithm: the StatArb strategy detailed in _Learn Algorithmic Trading_ by Sebastien Donadio and Sourav Ghosh was very helpful.
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - scipy
  - sklearn
