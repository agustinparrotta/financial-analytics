
import statistics as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle


class PairsTradingStrategy():
    def __init__(self, TRADING_INSTRUMENT, SYMBOLS, assets_data: dict):
        self.TRADING_INSTRUMENT = TRADING_INSTRUMENT
        self.SYMBOLS = SYMBOLS
        self.assets_data = assets_data
        self.delta_projected_actual_data = pd.DataFrame()


    def init_parameters(self, num_days, SMA_NUM_PERIODS=5, PRICE_DEV_NUM_PRICES=50):
        # Constantes utilizadas para computar el SMA (Simple Moving Average) y las deviaciones de precio de la misma.
        self.SMA_NUM_PERIODS = 5 # Período de look back
        self.price_history = {} # Historial de precios
        self.PRICE_DEV_NUM_PRICES = 50 # Período de look back period de las desviaciones de precio respecto a SMA
        self.price_deviation_from_sma = {} # Historial de las desviaciones de precio respecto a SMA


        self.num_days = num_days # Usado para iterar en todos los datos disponibles
        self.correlation_history = {} # Historial de correlaciones por par (BTC/ETH, BTC/GBTC)
        self.delta_projected_actual_history = {} # Historial de las diferencias entre desviaciones de Projected Prices y actual desviación de Price deviation por par
        self.final_delta_projected_history = [] # Historial final de diferencias entre desviaciones de Projected Prices y actual desviación de Price deviation por par


    def init_thresholds(self, BUY_ENTRY=10, SELL_ENTRY=-10, MIN_PRICE_MOVE_FROM_LAST_TRADE=100, NUM_SHARES_PER_TRADE=1,MIN_PROFIT_TO_CLOSE=10):
        # Variables de trade, position y pnl (P&L):
        self.orders = [] # Trackeo de órdenes de compra/venta, +1 para una orden de compra, -1 para orden de venta, 0 para no accion
        self.positions = [] # Trackeo de las posiciones, positivo para longs, negativo para shorts, 0 para no posición
        self.pnls = [] # Trackeo de pnl (la suma de closed_pnl)

        self.last_buy_price = 0 # Precio del último buy trade realizado, utilizado para prevenir over-trading al mismo precio
        self.last_sell_price = 0 # Precio del último sell trade realizado, utilizado para prevenir over-trading al mismo precio
        self.position = 0 # Posición actual de la estrategia
        self.buy_sum_price_qty = 0 # Suma de productos de buy_trade_price and buy_trade_qty por cada buy trade realizado al momento
        self.buy_sum_qty = 0 # Suma de buy_trade_qty por cada buy trade realizado al momento
        self.sell_sum_price_qty = 0 # Suma de productos de sell_trade_price and sell_trade_qty por cada sell trade realizado al momento
        self.sell_sum_qty = 0 # Suma de sell_trade_qty por cada sell trade realizado al momento
        self.open_pnl = 0 # Open PnL (no realizado)
        self.closed_pnl = 0 # Closed PnL hasta el momento (realizados)

        # Constantes de la estrategy (behavior y thresholds)
        self.StatArb_VALUE_FOR_BUY_ENTRY = BUY_ENTRY # Señal de trading StatArb sobre el cual ingresar buy-orders/long-position
        self.StatArb_VALUE_FOR_SELL_ENTRY = SELL_ENTRY # Señal de trading StatArb por debajo de la cual ingresar sell-orders/short-position
        self.MIN_PRICE_MOVE_FROM_LAST_TRADE = MIN_PRICE_MOVE_FROM_LAST_TRADE # Variación mínima de precio desde el último trade antes de realizar un nuevo trade (esto es para prevenir over-trading al mismo precio)
        self.NUM_SHARES_PER_TRADE = NUM_SHARES_PER_TRADE # Número de monedas a comprar/vender en cada trade. Esto no es algo que se quiera aprender en esta estrategia (en este caso solo aprendemos el side)
        self.MIN_PROFIT_TO_CLOSE = MIN_PROFIT_TO_CLOSE # Mínima ganancia Open para la cual cerrar posiciones y tomar ganancia


    def run_strategy(self, column='Close'):
        self.column = column

        for i in range(0, self.num_days):
            close_prices = {}

            # A partir de las series de precio, se calcula el SMA y sus desviaciones para cada activo.
            for symbol in self.SYMBOLS:
                close_prices[symbol] = self.assets_data[symbol][self.column].iloc[i]
                if not symbol in self.price_history.keys():
                    self.price_history[symbol] = []
                    self.price_deviation_from_sma[symbol] = []

                self.price_history[symbol].append(close_prices[symbol])
                if len(self.price_history[symbol]) > self.SMA_NUM_PERIODS:  # Trackeamos hasta SMA_NUM_PERIODS
                    del (self.price_history[symbol][0])

                sma = stats.mean(self.price_history[symbol]) # Rolling SimpleMovingAverage
                self.price_deviation_from_sma[symbol].append(close_prices[symbol] - sma) # Desviacion de SMA
                if len(self.price_deviation_from_sma[symbol]) > self.PRICE_DEV_NUM_PRICES:
                    del (self.price_deviation_from_sma[symbol][0])



            # Se calculan la correlacion entre BTC y sus proxies y la desviaciones de precio (las variables projected proyectan de acuerdo a la correlación)
            projected_dev_from_sma_using = {}
            for symbol in self.SYMBOLS:
                if symbol == self.TRADING_INSTRUMENT:  # No se calcula la relación entre BTC mismo
                    continue

                correlation_label = self.TRADING_INSTRUMENT + '<-' + symbol
                if correlation_label not in self.correlation_history.keys(): # Primera entrada para popular el diccionario
                    self.correlation_history[correlation_label] = []
                    self.delta_projected_actual_history[correlation_label] = []

                if len(self.price_deviation_from_sma[symbol]) < 2: # Se necesitan al menos 2 observaciones para calcular correlación
                    self.correlation_history[correlation_label].append(0)
                    self.delta_projected_actual_history[correlation_label].append(0)
                    continue

                corr = np.corrcoef(self.price_deviation_from_sma[self.TRADING_INSTRUMENT], self.price_deviation_from_sma[symbol])
                cov = np.cov(self.price_deviation_from_sma[self.TRADING_INSTRUMENT], self.price_deviation_from_sma[symbol])
                corr_trading_instrument_lead_instrument = corr[0, 1]  # Correlación entre las 2 series
                cov_trading_instrument_lead_instrument = cov[0, 0] / cov[0, 1] # Covarianza entre las 2 series

                self.correlation_history[correlation_label].append(corr_trading_instrument_lead_instrument)

                # projected-price-deviation-in-TRADING_INSTRUMENT es covariance * price-deviation-in-lead-symbol
                projected_dev_from_sma_using[symbol] = self.price_deviation_from_sma[symbol][-1] * cov_trading_instrument_lead_instrument

                # delta +ve => señal indica que el precio de TRADING_INSTRUMENT debería haberse movido más de lo que lo hizo
                # delta -ve => signal indica que el precio de TRADING_INSTRUMENT debería haberse movido menos de lo que lo hizo
                delta_projected_actual = (projected_dev_from_sma_using[symbol] - self.price_deviation_from_sma[self.TRADING_INSTRUMENT][-1])
                self.delta_projected_actual_history[correlation_label].append(delta_projected_actual)


            # Se combinan los deltas anteriores para generar una única señal. Luego se normalizan los valores.

            # Weight para cada par, definido como la correlación entre ese par
            sum_weights = 0 # La suma de weights es la suma de correlaciones para para activo con BTC
            for symbol in self.SYMBOLS:
                if symbol == self.TRADING_INSTRUMENT:  # No se calcula la relación entre BTC mismo
                    continue

                correlation_label = self.TRADING_INSTRUMENT + '<-' + symbol
                sum_weights += abs(self.correlation_history[correlation_label][-1])

            final_delta_projected = 0
            close_price = close_prices[self.TRADING_INSTRUMENT]
            for symbol in self.SYMBOLS:
                if symbol == self.TRADING_INSTRUMENT:  # No se calcula la relación entre BTC mismo
                    continue

                correlation_label = self.TRADING_INSTRUMENT + '<-' + symbol

                # Proyección ponderada
                final_delta_projected += (abs(self.correlation_history[correlation_label][-1]) * self.delta_projected_actual_history[correlation_label][-1])

            # Normalización
            if sum_weights != 0:
                final_delta_projected /= sum_weights
            else:
                final_delta_projected = 0

            self.final_delta_projected_history.append(final_delta_projected)



            # Aplicación de la estrategia. Acá se chequea la señal de tradeo considerando los parámetros/thresholds y posiciones
        
            # Se realizará un sell trade al close_prices si se cumple alguna de las siguientes condiciones:
                # 1. La señal de trading esta por debajo de Sell-Entry threshold y la diferencia entre el último y el actual precio de trade es suficiente.
                # 2. Estamos long( +ve position ) y la posición actual es suficientemente buena (profitable) para tomar ganancia.
            if ((final_delta_projected < self.StatArb_VALUE_FOR_SELL_ENTRY and abs(close_price - self.last_sell_price) > self.MIN_PRICE_MOVE_FROM_LAST_TRADE)  # Señal por debajo del threshold, debemos vender
                or
                (self.position > 0 and (self.open_pnl > self.MIN_PROFIT_TO_CLOSE))):  # Estamos long de -ve StatArb y la señal se ha hechos positiva o la posición es buena, debemos vender para cerrar posición
                self.orders.append(-1)  # Marca de sell trade
                self.last_sell_price = close_price
                self.position -= self.NUM_SHARES_PER_TRADE  # Reducir posición por el tamaño de este trade
                self.sell_sum_price_qty += (close_price * self.NUM_SHARES_PER_TRADE)  # Actualizar suma de sell-price-qty
                self.sell_sum_qty += self.NUM_SHARES_PER_TRADE # Actualizar suma de sell-price
                print("Sell ", self.NUM_SHARES_PER_TRADE, " @ ", close_price, "Position: ", self.position)
                print("OpenPnL: ", self.open_pnl, " ClosedPnL: ", self.closed_pnl, " TotalPnL: ", (self.open_pnl + self.closed_pnl))

            # Se realizará un buy trade al close_prices si se cumple alguna de las siguientes condiciones:
                # 1. La señal de trading esta por encima de Buy-Entry threshold y la diferencia entre el último y el actual precio de trade es suficiente.
                # 2. Estamos short( -ve position ) y la posición actual es suficientemente buena (profitable) para tomar ganancia.
            elif ((final_delta_projected > self.StatArb_VALUE_FOR_BUY_ENTRY and abs(close_price - self.last_buy_price) > self.MIN_PRICE_MOVE_FROM_LAST_TRADE)  # Señal por encima del threshold, debemos comprar
                or
                (self.position < 0 and (self.open_pnl > self.MIN_PROFIT_TO_CLOSE))):  # Estamos short de +ve StatArb y la señal se ha hecho negativa o la posición es buena, debemos comprar para cerrar posición
                self.orders.append(+1)  # Marca de buy trade
                self.last_buy_price = close_price
                self.position += self.NUM_SHARES_PER_TRADE  # Aumentar posición por el tamaño de este trade
                self.buy_sum_price_qty += (close_price * self.NUM_SHARES_PER_TRADE)  # Actualizar suma de buy-price-qty
                self.buy_sum_qty += self.NUM_SHARES_PER_TRADE # Actualizar suma de buy-price
                print("Buy ", self.NUM_SHARES_PER_TRADE, " @ ", close_price, "Position: ", self.position)
                print("OpenPnL: ", self.open_pnl, " ClosedPnL: ", self.closed_pnl, " TotalPnL: ", (self.open_pnl + self.closed_pnl))
            else:
            # No hay trade ya que ninguna condición se cumplió para comprar o vender
                self.orders.append(0)

            self.positions.append(self.position)



            # Esta sección actualiza las posiciones Open/Unrealized, Closed/Realized y el pnl
            self.open_pnl = 0
            if self.position > 0:
                if self.sell_sum_qty > 0:  # long position and some sell trades have been made against it, close that amount based on how much was sold against this long position
                    self.open_pnl = abs(self.sell_sum_qty) * (self.sell_sum_price_qty / self.sell_sum_qty - self.buy_sum_price_qty / self.buy_sum_qty)
                # mark the remaining position to market i.e. pnl would be what it would be if we closed at current price
                self.open_pnl += abs(self.sell_sum_qty - self.position) * (close_price - self.buy_sum_price_qty / self.buy_sum_qty)
            elif self.position < 0:
                if self.buy_sum_qty > 0:  # short position and some buy trades have been made against it, close that amount based on how much was bought against this short position
                    self.open_pnl = abs(self.buy_sum_qty) * (self.sell_sum_price_qty / self.sell_sum_qty - self.buy_sum_price_qty / self.buy_sum_qty)
                # mark the remaining position to market i.e. pnl would be what it would be if we closed at current price
                self.open_pnl += abs(self.buy_sum_qty - self.position) * (self.sell_sum_price_qty / self.sell_sum_qty - close_price)
            else:
                # flat, so update closed_pnl and reset tracking variables for positions & pnls
                self.closed_pnl += (self.sell_sum_price_qty - self.buy_sum_price_qty)
                self.buy_sum_price_qty = 0
                self.buy_sum_qty = 0
                self.sell_sum_price_qty = 0
                self.sell_sum_qty = 0
                self.last_buy_price = 0
                self.last_sell_price = 0

            self.pnls.append(self.closed_pnl + self.open_pnl)


    def plot_correlations(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10), dpi=120)
        cycol = cycle('bgrcmky')
        correlation_data = pd.DataFrame()
        for symbol in self.SYMBOLS:
            if symbol == self.TRADING_INSTRUMENT:
                continue
            correlation_label = self.TRADING_INSTRUMENT + '<-' + symbol
            correlation_data = correlation_data.assign(label=pd.Series(self.correlation_history[correlation_label], index=self.assets_data[symbol].index))
            ax = correlation_data['label'].plot(color=next(cycol), lw=1., label='Correlation ' + correlation_label)
        for i in np.arange(-1, 1, 0.25):
            plt.axhline(y=i, lw=0.5, color='k')
        plt.legend()
        plt.show()


    def plot_trading_signal(self):
        # Plot StatArb signal provided by each currency pair
        plt.figure(figsize=(20, 10), dpi=120)
        cycol = cycle('bgrcmky')
        
        for symbol in self.SYMBOLS:
            if symbol == self.TRADING_INSTRUMENT:
                continue

            projection_label = self.TRADING_INSTRUMENT + '<-' + symbol
            self.delta_projected_actual_data = self.delta_projected_actual_data.assign(StatArbTradingSignal=pd.Series(self.delta_projected_actual_history[projection_label], index=self.assets_data[self.TRADING_INSTRUMENT].index))
            ax = self.delta_projected_actual_data['StatArbTradingSignal'].plot(color=next(cycol), lw=1., label='StatArbTradingSignal ' + projection_label)
        plt.legend()
        plt.show()


    def plot_prices_with_signals(self):
        plt.figure(figsize=(30, 20), dpi=120)
        self.delta_projected_actual_data = pd.DataFrame()
        self.delta_projected_actual_data = self.delta_projected_actual_data.assign(ClosePrice=pd.Series(self.assets_data[self.TRADING_INSTRUMENT][self.column], index=self.assets_data[self.TRADING_INSTRUMENT].index))
        self.delta_projected_actual_data = self.delta_projected_actual_data.assign(FinalStatArbTradingSignal=pd.Series(self.final_delta_projected_history, index=self.assets_data[self.TRADING_INSTRUMENT].index))
        self.delta_projected_actual_data = self.delta_projected_actual_data.assign(Trades=pd.Series(self.orders, index=self.assets_data[self.TRADING_INSTRUMENT].index))
        self.delta_projected_actual_data = self.delta_projected_actual_data.assign(Position=pd.Series(self.positions, index=self.assets_data[self.TRADING_INSTRUMENT].index))
        self.delta_projected_actual_data = self.delta_projected_actual_data.assign(Pnl=pd.Series(self.pnls, index=self.assets_data[self.TRADING_INSTRUMENT].index))

        plt.plot(self.delta_projected_actual_data.index, self.delta_projected_actual_data.ClosePrice, color='k', lw=1., label='ClosePrice')
        plt.plot(self.delta_projected_actual_data.loc[self.delta_projected_actual_data.Trades == 1].index, self.delta_projected_actual_data.ClosePrice[self.delta_projected_actual_data.Trades == 1], color='r', lw=0, marker='^', markersize=7, label='buy')
        plt.plot(self.delta_projected_actual_data.loc[self.delta_projected_actual_data.Trades == -1].index, self.delta_projected_actual_data.ClosePrice[self.delta_projected_actual_data.Trades == -1], color='g', lw=0, marker='v', markersize=7, label='sell')
        plt.legend()
        plt.show()

    def plot_final_trading_signal(self):
        plt.figure(figsize=(30, 20), dpi=120)
        plt.plot(self.delta_projected_actual_data.index, self.delta_projected_actual_data.FinalStatArbTradingSignal, color='k', lw=1., label='FinalStatArbTradingSignal')
        plt.plot(self.delta_projected_actual_data.loc[self.delta_projected_actual_data.Trades == 1].index, self.delta_projected_actual_data.FinalStatArbTradingSignal[self.delta_projected_actual_data.Trades == 1], color='r', lw=0, marker='^', markersize=7, label='buy')
        plt.plot(self.delta_projected_actual_data.loc[self.delta_projected_actual_data.Trades == -1].index, self.delta_projected_actual_data.FinalStatArbTradingSignal[self.delta_projected_actual_data.Trades == -1], color='g', lw=0, marker='v', markersize=7, label='sell')
        plt.axhline(y=0, lw=0.5, color='k')
        for i in np.arange(self.StatArb_VALUE_FOR_BUY_ENTRY, self.StatArb_VALUE_FOR_BUY_ENTRY * 10, self.StatArb_VALUE_FOR_BUY_ENTRY * 2):
            plt.axhline(y=i, lw=0.5, color='r')
        for i in np.arange(self.StatArb_VALUE_FOR_SELL_ENTRY, self.StatArb_VALUE_FOR_SELL_ENTRY * 10, self.StatArb_VALUE_FOR_SELL_ENTRY * 2):
            plt.axhline(y=i, lw=0.5, color='g')
        plt.legend()
        plt.show()

    def plot_positions(self):
        plt.figure(figsize=(30, 20), dpi=120)
        plt.plot(self.delta_projected_actual_data.index, self.delta_projected_actual_data.Position, color='k', lw=1., label='Position')
        plt.plot(self.delta_projected_actual_data.loc[self.delta_projected_actual_data.Position == 0].index, self.delta_projected_actual_data.Position[self.delta_projected_actual_data.Position == 0], color='k', lw=0, marker='.', label='flat')
        plt.plot(self.delta_projected_actual_data.loc[self.delta_projected_actual_data.Position > 0].index, self.delta_projected_actual_data.Position[self.delta_projected_actual_data.Position > 0], color='r', lw=0, marker='+', label='long')
        plt.plot(self.delta_projected_actual_data.loc[self.delta_projected_actual_data.Position < 0].index, self.delta_projected_actual_data.Position[self.delta_projected_actual_data.Position < 0], color='g', lw=0, marker='_', label='short')
        plt.axhline(y=0, lw=0.5, color='k')
        for i in range(self.NUM_SHARES_PER_TRADE, self.NUM_SHARES_PER_TRADE * 5, self.NUM_SHARES_PER_TRADE):
            plt.axhline(y=i, lw=0.5, color='r')
        for i in range(-self.NUM_SHARES_PER_TRADE, -self.NUM_SHARES_PER_TRADE * 5, -self.NUM_SHARES_PER_TRADE):
            plt.axhline(y=i, lw=0.5, color='g')
        plt.legend()
        plt.show()

    def plot_pnl(self):
        plt.figure(figsize=(30, 20), dpi=120)
        plt.plot(self.delta_projected_actual_data.index, self.delta_projected_actual_data.Pnl, color='k', lw=1., label='Pnl')
        plt.plot(self.delta_projected_actual_data.loc[self.delta_projected_actual_data.Pnl > 0].index, self.delta_projected_actual_data.Pnl[self.delta_projected_actual_data.Pnl > 0], color='g', lw=0, marker='.')
        plt.plot(self.delta_projected_actual_data.loc[self.delta_projected_actual_data.Pnl < 0].index, self.delta_projected_actual_data.Pnl[self.delta_projected_actual_data.Pnl < 0], color='r', lw=0, marker='.')
        plt.legend()
        plt.show()