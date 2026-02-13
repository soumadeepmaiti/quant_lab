"""
Execution algorithms: VWAP, TWAP, POV.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from quantlab.microstructure.lob import LimitOrderBook


@dataclass
class ExecutionResult:
    """Container for execution results."""
    total_size: int
    executed_size: int
    n_slices: int
    vwap: float
    initial_mid: float
    final_mid: float
    slippage: float
    slippage_bps: float
    slice_vwaps: List[float]
    trades: list


class VWAPExecutor:
    """
    Volume Weighted Average Price execution algorithm.
    
    Slices large orders and executes over time to minimize impact.
    """

    def __init__(self, order_book: LimitOrderBook):
        self.book = order_book
        self.execution_log = []

    def execute(
        self,
        size: int,
        side: str,
        n_slices: int = 10,
        simulate_activity: bool = True,
        activity_orders: int = 30
    ) -> ExecutionResult:
        """
        Execute a VWAP order.
        
        Parameters
        ----------
        size : int
            Total order size
        side : str
            'BUY' or 'SELL'
        n_slices : int
            Number of execution slices
        simulate_activity : bool
            Simulate market activity between slices
        activity_orders : int
            Random orders between slices
        
        Returns
        -------
        ExecutionResult
            Execution summary
        """
        initial_mid = self.book.get_mid_price()

        base_slice = size // n_slices
        remainder = size % n_slices

        all_trades = []
        slice_vwaps = []
        total_executed = 0
        total_value = 0

        for i in range(n_slices):
            slice_size = base_slice + (1 if i < remainder else 0)

            trades, slice_vwap = self.book.execute_market_order(side, slice_size)

            slice_qty = sum(t.quantity for t in trades)
            slice_value = sum(t.price * t.quantity for t in trades)

            all_trades.extend(trades)
            total_executed += slice_qty
            total_value += slice_value
            slice_vwaps.append(slice_vwap)

            self.execution_log.append({
                'slice': i + 1,
                'size': slice_size,
                'executed': slice_qty,
                'vwap': slice_vwap
            })

            # Simulate market activity
            if simulate_activity and i < n_slices - 1:
                self._simulate_activity(activity_orders)

        overall_vwap = total_value / total_executed if total_executed > 0 else 0
        final_mid = self.book.get_mid_price()

        if side == 'BUY':
            slippage = overall_vwap - initial_mid if initial_mid else 0
        else:
            slippage = initial_mid - overall_vwap if initial_mid else 0

        slippage_bps = (slippage / initial_mid * 10000) if initial_mid else 0

        return ExecutionResult(
            total_size=size,
            executed_size=total_executed,
            n_slices=n_slices,
            vwap=overall_vwap,
            initial_mid=initial_mid,
            final_mid=final_mid,
            slippage=slippage,
            slippage_bps=slippage_bps,
            slice_vwaps=slice_vwaps,
            trades=all_trades
        )

    def _simulate_activity(self, n_orders: int):
        """Simulate random market activity."""
        for _ in range(n_orders):
            side = 'BUY' if np.random.random() < 0.5 else 'SELL'
            size = int(np.random.lognormal(5, 0.5))

            mid = self.book.get_mid_price() or 100.0
            offset = np.random.uniform(0.01, 0.05)

            if side == 'BUY':
                price = round(mid - offset, 2)
            else:
                price = round(mid + offset, 2)

            if np.random.random() < 0.8:
                self.book.add_limit_order(side, price, size)
            else:
                self.book.execute_market_order(side, min(size, 200))


class TWAPExecutor:
    """
    Time Weighted Average Price execution algorithm.
    
    Executes equal-sized slices at fixed time intervals.
    """

    def __init__(self, order_book: LimitOrderBook):
        self.book = order_book
        self.execution_log = []

    def execute(
        self,
        size: int,
        side: str,
        n_slices: int = 10,
        simulate_activity: bool = True,
        activity_orders: int = 30
    ) -> ExecutionResult:
        """
        Execute a TWAP order.
        
        Equal-sized slices regardless of volume profile.
        """
        initial_mid = self.book.get_mid_price()

        slice_size = size // n_slices
        remainder = size % n_slices

        all_trades = []
        slice_vwaps = []
        total_executed = 0
        total_value = 0

        for i in range(n_slices):
            current_slice = slice_size + (1 if i < remainder else 0)

            trades, slice_vwap = self.book.execute_market_order(side, current_slice)

            slice_qty = sum(t.quantity for t in trades)
            slice_value = sum(t.price * t.quantity for t in trades)

            all_trades.extend(trades)
            total_executed += slice_qty
            total_value += slice_value
            slice_vwaps.append(slice_vwap)

            self.execution_log.append({
                'slice': i + 1,
                'size': current_slice,
                'executed': slice_qty,
                'vwap': slice_vwap
            })

            if simulate_activity and i < n_slices - 1:
                self._simulate_activity(activity_orders)

        overall_vwap = total_value / total_executed if total_executed > 0 else 0
        final_mid = self.book.get_mid_price()

        if side == 'BUY':
            slippage = overall_vwap - initial_mid if initial_mid else 0
        else:
            slippage = initial_mid - overall_vwap if initial_mid else 0

        slippage_bps = (slippage / initial_mid * 10000) if initial_mid else 0

        return ExecutionResult(
            total_size=size,
            executed_size=total_executed,
            n_slices=n_slices,
            vwap=overall_vwap,
            initial_mid=initial_mid,
            final_mid=final_mid,
            slippage=slippage,
            slippage_bps=slippage_bps,
            slice_vwaps=slice_vwaps,
            trades=all_trades
        )

    def _simulate_activity(self, n_orders: int):
        """Simulate random market activity."""
        for _ in range(n_orders):
            side = 'BUY' if np.random.random() < 0.5 else 'SELL'
            size = int(np.random.lognormal(5, 0.5))

            mid = self.book.get_mid_price() or 100.0
            offset = np.random.uniform(0.01, 0.05)

            price = round(mid - offset, 2) if side == 'BUY' else round(mid + offset, 2)

            if np.random.random() < 0.8:
                self.book.add_limit_order(side, price, size)
            else:
                self.book.execute_market_order(side, min(size, 200))


class POVExecutor:
    """
    Percentage of Volume execution algorithm.
    
    Executes as a fixed percentage of market volume.
    """

    def __init__(self, order_book: LimitOrderBook):
        self.book = order_book
        self.execution_log = []

    def execute(
        self,
        size: int,
        side: str,
        participation_rate: float = 0.1,
        max_slices: int = 20,
        simulate_activity: bool = True,
        activity_orders: int = 50
    ) -> ExecutionResult:
        """
        Execute a POV order.
        
        Parameters
        ----------
        size : int
            Total order size
        side : str
            'BUY' or 'SELL'
        participation_rate : float
            Target percentage of volume (0.1 = 10%)
        max_slices : int
            Maximum number of slices
        simulate_activity : bool
            Simulate market activity
        activity_orders : int
            Activity between slices
        """
        initial_mid = self.book.get_mid_price()

        all_trades = []
        slice_vwaps = []
        total_executed = 0
        total_value = 0
        remaining = size
        n_slices = 0

        while remaining > 0 and n_slices < max_slices:
            # Simulate market volume
            market_volume = activity_orders * 100  # Rough estimate
            slice_size = min(int(market_volume * participation_rate), remaining)
            slice_size = max(slice_size, 100)  # Minimum slice

            trades, slice_vwap = self.book.execute_market_order(side, slice_size)

            slice_qty = sum(t.quantity for t in trades)
            slice_value = sum(t.price * t.quantity for t in trades)

            all_trades.extend(trades)
            total_executed += slice_qty
            total_value += slice_value
            slice_vwaps.append(slice_vwap)
            remaining -= slice_qty
            n_slices += 1

            self.execution_log.append({
                'slice': n_slices,
                'size': slice_size,
                'executed': slice_qty,
                'vwap': slice_vwap
            })

            if simulate_activity and remaining > 0:
                self._simulate_activity(activity_orders)

        overall_vwap = total_value / total_executed if total_executed > 0 else 0
        final_mid = self.book.get_mid_price()

        if side == 'BUY':
            slippage = overall_vwap - initial_mid if initial_mid else 0
        else:
            slippage = initial_mid - overall_vwap if initial_mid else 0

        slippage_bps = (slippage / initial_mid * 10000) if initial_mid else 0

        return ExecutionResult(
            total_size=size,
            executed_size=total_executed,
            n_slices=n_slices,
            vwap=overall_vwap,
            initial_mid=initial_mid,
            final_mid=final_mid,
            slippage=slippage,
            slippage_bps=slippage_bps,
            slice_vwaps=slice_vwaps,
            trades=all_trades
        )

    def _simulate_activity(self, n_orders: int):
        """Simulate market activity."""
        for _ in range(n_orders):
            side = 'BUY' if np.random.random() < 0.5 else 'SELL'
            size = int(np.random.lognormal(5, 0.5))

            mid = self.book.get_mid_price() or 100.0
            offset = np.random.uniform(0.01, 0.05)

            price = round(mid - offset, 2) if side == 'BUY' else round(mid + offset, 2)

            if np.random.random() < 0.8:
                self.book.add_limit_order(side, price, size)
            else:
                self.book.execute_market_order(side, min(size, 200))


def compare_strategies(
    book_factory,
    size: int,
    side: str = 'BUY',
    n_slices: int = 10
) -> pd.DataFrame:
    """
    Compare execution strategies.
    
    Parameters
    ----------
    book_factory : callable
        Function that creates a fresh order book
    size : int
        Order size
    side : str
        Order side
    n_slices : int
        Number of slices
    
    Returns
    -------
    pd.DataFrame
        Comparison of strategies
    """
    results = []

    # Aggressive (single market order)
    book = book_factory()
    from quantlab.microstructure.impact import analyze_market_order
    aggressive = analyze_market_order(book, size, side)
    results.append({
        'strategy': 'Aggressive',
        'slippage_bps': aggressive['slippage_bps'],
        'vwap': aggressive['vwap']
    })

    # VWAP
    book = book_factory()
    vwap_exec = VWAPExecutor(book)
    vwap_result = vwap_exec.execute(size, side, n_slices)
    results.append({
        'strategy': 'VWAP',
        'slippage_bps': vwap_result.slippage_bps,
        'vwap': vwap_result.vwap
    })

    # TWAP
    book = book_factory()
    twap_exec = TWAPExecutor(book)
    twap_result = twap_exec.execute(size, side, n_slices)
    results.append({
        'strategy': 'TWAP',
        'slippage_bps': twap_result.slippage_bps,
        'vwap': twap_result.vwap
    })

    df = pd.DataFrame(results)
    df['savings_vs_aggressive'] = df.iloc[0]['slippage_bps'] - df['slippage_bps']

    return df
