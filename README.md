# Financial Analytics

Se aborda una estrategia de _pairs trading_ utilizando Bitcoin y sus proxies. Estas estrategias buscan explotar diferencias de corto plazo entre activos altamente correlacionados, tomando una posición _long_ en el activo que está underperforming y _short_ en el overperforming. Se analizan distintos proxies de Bitcoin:

- Criptomonedas: Ethereum (ETH)
- Microstrategy (MSTR)
- Grayscale Bitcoin Investment Trust (GBTC)
- Empresas de minería de Bitcoin: MARA, RIOT
- ETFs: BITO, BITW

Sin embargo, se termina utilizando GBTC como único proxy debido a su alta correlación.

### Recursos:

- Fuente de datos: Coin Metrics para la serie de precios de criptomonedas y Yahoo Finance para el resto.
- Código fuente: nos apoyamos en los códigos proporcionados en el libro _Advances in Financial Machine Learning_, de Marcos López de Prado.
- Algoritmo de _pairs trading_: la estrategia StatArb detallada en Learn Algorithmic Trading, de Sebastien Donadio y Sourav Ghosh, fue de gran ayuda.
- Librerías:
  - pandas
  - numpy
  - matplotlib
  - scipy
  - sklearn
