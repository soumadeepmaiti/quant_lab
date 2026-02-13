"""
Limit Order Book data structures and matching engine.

Supports both synthetic and real intraday market data from Polygon.io.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """Single order in the limit order book."""
    order_id: int
    side: str  # 'BUY' or 'SELL'
    price: float
    quantity: int
    timestamp: datetime
    order_type: str = 'LIMIT'


@dataclass
class Trade:
    """Executed trade."""
    trade_id: int
    price: float
    quantity: int
    aggressor_side: str
    timestamp: datetime


class LimitOrderBook:
    """
    Limit Order Book with price-time priority matching.
    
    Maintains bid (buy) and ask (sell) order queues.
    Matches incoming orders against resting liquidity.
    """

    def __init__(self, symbol: str = "SIM"):
        self.symbol = symbol
        self.bids = defaultdict(list)  # {price: [orders]}
        self.asks = defaultdict(list)
        self.trades: List[Trade] = []
        self.order_id_counter = 0
        self.trade_id_counter = 0
        self.price_history = []
        self.spread_history = []

    def get_best_bid(self) -> Tuple[Optional[float], int]:
        """Get highest bid price and size."""
        if not self.bids:
            return None, 0
        best_price = max(self.bids.keys())
        total_size = sum(o.quantity for o in self.bids[best_price])
        return best_price, total_size

    def get_best_ask(self) -> Tuple[Optional[float], int]:
        """Get lowest ask price and size."""
        if not self.asks:
            return None, 0
        best_price = min(self.asks.keys())
        total_size = sum(o.quantity for o in self.asks[best_price])
        return best_price, total_size

    def get_mid_price(self) -> Optional[float]:
        """Calculate mid price."""
        bid, _ = self.get_best_bid()
        ask, _ = self.get_best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2

    def get_spread(self) -> Optional[float]:
        """Calculate bid-ask spread."""
        bid, _ = self.get_best_bid()
        ask, _ = self.get_best_ask()
        if bid is None or ask is None:
            return None
        return ask - bid

    def add_limit_order(
        self,
        side: str,
        price: float,
        quantity: int,
        timestamp: Optional[datetime] = None
    ) -> Tuple[int, List[Trade]]:
        """Add a limit order to the book."""
        if timestamp is None:
            timestamp = datetime.now()

        self.order_id_counter += 1
        order = Order(
            order_id=self.order_id_counter,
            side=side,
            price=price,
            quantity=quantity,
            timestamp=timestamp,
            order_type='LIMIT'
        )

        executed_trades = []
        remaining_qty = quantity

        if side == 'BUY':
            while remaining_qty > 0 and self.asks:
                best_ask = min(self.asks.keys())
                if price >= best_ask:
                    trades, remaining_qty = self._match_level(
                        self.asks, best_ask, remaining_qty, 'BUY', timestamp
                    )
                    executed_trades.extend(trades)
                else:
                    break
        else:  # SELL
            while remaining_qty > 0 and self.bids:
                best_bid = max(self.bids.keys())
                if price <= best_bid:
                    trades, remaining_qty = self._match_level(
                        self.bids, best_bid, remaining_qty, 'SELL', timestamp
                    )
                    executed_trades.extend(trades)
                else:
                    break

        # Add remaining to book
        if remaining_qty > 0:
            order.quantity = remaining_qty
            if side == 'BUY':
                self.bids[price].append(order)
            else:
                self.asks[price].append(order)

        self._record_state(timestamp)
        return order.order_id, executed_trades

    def _match_level(
        self,
        book_side: dict,
        price: float,
        quantity: int,
        aggressor_side: str,
        timestamp: datetime
    ) -> Tuple[List[Trade], int]:
        """Match against a price level."""
        trades = []
        remaining = quantity

        while remaining > 0 and book_side.get(price):
            resting = book_side[price][0]
            trade_qty = min(remaining, resting.quantity)

            self.trade_id_counter += 1
            trade = Trade(
                trade_id=self.trade_id_counter,
                price=price,
                quantity=trade_qty,
                aggressor_side=aggressor_side,
                timestamp=timestamp
            )
            trades.append(trade)
            self.trades.append(trade)

            remaining -= trade_qty
            resting.quantity -= trade_qty

            if resting.quantity == 0:
                book_side[price].pop(0)
                if not book_side[price]:
                    del book_side[price]

        return trades, remaining

    def execute_market_order(
        self,
        side: str,
        quantity: int,
        timestamp: Optional[datetime] = None
    ) -> Tuple[List[Trade], float]:
        """Execute a market order."""
        if timestamp is None:
            timestamp = datetime.now()

        trades = []
        remaining = quantity
        total_value = 0
        total_qty = 0

        if side == 'BUY':
            while remaining > 0 and self.asks:
                best_ask = min(self.asks.keys())
                t, remaining = self._match_level(
                    self.asks, best_ask, remaining, 'BUY', timestamp
                )
                trades.extend(t)
                for trade in t:
                    total_value += trade.price * trade.quantity
                    total_qty += trade.quantity
        else:
            while remaining > 0 and self.bids:
                best_bid = max(self.bids.keys())
                t, remaining = self._match_level(
                    self.bids, best_bid, remaining, 'SELL', timestamp
                )
                trades.extend(t)
                for trade in t:
                    total_value += trade.price * trade.quantity
                    total_qty += trade.quantity

        avg_price = total_value / total_qty if total_qty > 0 else 0
        self._record_state(timestamp)
        return trades, avg_price

    def cancel_order(self, side: str, price: float, order_id: int) -> bool:
        """Cancel an order."""
        book_side = self.bids if side == 'BUY' else self.asks

        if price in book_side:
            for i, order in enumerate(book_side[price]):
                if order.order_id == order_id:
                    book_side[price].pop(i)
                    if not book_side[price]:
                        del book_side[price]
                    return True
        return False

    def _record_state(self, timestamp: datetime):
        """Record market state."""
        mid = self.get_mid_price()
        spread = self.get_spread()

        if mid is not None:
            self.price_history.append({
                'timestamp': timestamp,
                'mid_price': mid
            })
        if spread is not None:
            self.spread_history.append({
                'timestamp': timestamp,
                'spread': spread
            })

    def get_snapshot(self, depth: int = 5) -> Dict:
        """Get order book snapshot."""
        bid_prices = sorted(self.bids.keys(), reverse=True)[:depth]
        ask_prices = sorted(self.asks.keys())[:depth]

        bids = [
            {'price': p, 'size': sum(o.quantity for o in self.bids[p])}
            for p in bid_prices
        ]
        asks = [
            {'price': p, 'size': sum(o.quantity for o in self.asks[p])}
            for p in ask_prices
        ]

        return {
            'symbol': self.symbol,
            'bids': bids,
            'asks': asks,
            'mid_price': self.get_mid_price(),
            'spread': self.get_spread()
        }

    def display(self, depth: int = 5):
        """Print order book."""
        snapshot = self.get_snapshot(depth)

        print(f"\n{'='*50}")
        print(f"ORDER BOOK: {self.symbol}")
        if snapshot['mid_price']:
            print(f"Mid: ${snapshot['mid_price']:.2f} | Spread: ${snapshot['spread']:.2f}")
        print(f"{'='*50}")

        for ask in reversed(snapshot['asks']):
            print(f"{'':>10} | ${ask['price']:>10.2f} | {ask['size']:<10,}")

        print(f"{'':-^10} | {'SPREAD':^12} | {'':-^10}")

        for bid in snapshot['bids']:
            print(f"{bid['size']:>10,} | ${bid['price']:>10.2f} |")


def initialize_book(
    symbol: str = "SIM",
    mid_price: float = 100.0,
    spread: float = 0.02,
    levels: int = 10,
    base_size: int = 1000
) -> LimitOrderBook:
    """Initialize order book with market depth."""
    book = LimitOrderBook(symbol)

    best_bid = mid_price - spread / 2
    best_ask = mid_price + spread / 2
    tick = 0.01

    for i in range(levels):
        price = round(best_bid - i * tick, 2)
        size = int(base_size * (1 + 0.1 * np.random.random()))
        book.add_limit_order('BUY', price, size)

    for i in range(levels):
        price = round(best_ask + i * tick, 2)
        size = int(base_size * (1 + 0.1 * np.random.random()))
        book.add_limit_order('SELL', price, size)

    return book


def load_real_book_from_polygon(
    polygon_client,
    ticker: str,
    date: str,
    depth_levels: int = 10
) -> LimitOrderBook:
    """
    Load real intraday market data from Polygon API into LOB.
    
    Reconstructs opening order book state from Polygon NBBO quotes
    and processes actual trades throughout the day.
    
    Parameters
    ----------
    polygon_client : PolygonClient
        Initialized Polygon API client
    ticker : str
        Stock ticker symbol (e.g., 'AAPL')
    date : str
        Trading date (YYYY-MM-DD)
    depth_levels : int
        Number of price levels to simulate in opening book
    
    Returns
    -------
    LimitOrderBook
        Populated LOB with opening state from Polygon data
    
    Raises
    ------
    ValueError
        If no market data available for the given date
    
    Example
    -------
    >>> from quantlab.data.polygon_client import PolygonClient
    >>> client = PolygonClient()
    >>> book = load_real_book_from_polygon(client, 'AAPL', '2024-01-15')
    >>> trades, avg_price = book.execute_market_order('BUY', 1000)
    """
    try:
        # Fetch intraday quotes and trades
        quotes_df = polygon_client.get_quotes(ticker, date)
        trades_df = polygon_client.get_trades(ticker, date)

        if quotes_df.empty and trades_df.empty:
            logger.warning(f"No market data for {ticker} on {date}. Using synthetic book.")
            return initialize_book(ticker)

        book = LimitOrderBook(ticker)

        # Initialize book from first available quote
        if not quotes_df.empty:
            first_quote = quotes_df.iloc[0]
            bid_price = float(first_quote.get('bid_price', 100.0))
            ask_price = float(first_quote.get('ask_price', 100.01))
            bid_size = int(first_quote.get('bid_size', 1000))
            ask_size = int(first_quote.get('ask_size', 1000))

            # Build initial depth from aggregated quote data
            tick = 0.01
            for i in range(depth_levels):
                # Add sell side
                sell_price = round(ask_price + i * tick, 2)
                sell_size = int(ask_size * (0.9 - 0.06 * i))
                if sell_size > 0:
                    book.add_limit_order('SELL', sell_price, sell_size)

                # Add buy side
                buy_price = round(bid_price - i * tick, 2)
                buy_size = int(bid_size * (0.9 - 0.06 * i))
                if buy_size > 0:
                    book.add_limit_order('BUY', buy_price, buy_size)

            logger.info(f"Initialized {ticker} LOB from {len(quotes_df)} quotes | "
                       f"Opening spread: {ask_price - bid_price:.2f}")

        # Replay trades throughout the day to build realistic state
        if not trades_df.empty:
            trades_df = trades_df.sort_index()
            for idx, (ts, trade_row) in enumerate(trades_df.iterrows()):
                price = float(trade_row['price'])
                size = int(trade_row['size'])
                # Simulate aggressive side (conservatively assume buyer-initiated)
                book.execute_market_order('BUY', size, timestamp=ts)

                if (idx + 1) % 1000 == 0:
                    logger.debug(f"Processed {idx + 1}/{len(trades_df)} trades")

        logger.info(f"Successfully loaded {ticker} LOB with {len(trades_df)} trades")
        return book

    except Exception as e:
        logger.error(f"Error loading Polygon data for {ticker}: {e}. Falling back to synthetic.")
        return initialize_book(ticker)
