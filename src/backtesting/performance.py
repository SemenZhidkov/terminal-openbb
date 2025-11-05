"""
Performance Analyzer
Расчет метрик эффективности торговой стратегии
"""

import logging
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from .portfolio import Portfolio

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Анализатор производительности стратегии
    
    Метрики:
    - Sharpe Ratio
    - Sortino Ratio
    - Calmar Ratio
    - Maximum Drawdown
    - Win Rate
    - Profit Factor
    - Average Trade
    - Recovery Factor
    - Ulcer Index
    - Monthly/Annual Returns
    """
    
    def __init__(self, portfolio: Portfolio, risk_free_rate: float = 0.02):
        """
        Инициализация
        
        Args:
            portfolio: Портфель для анализа
            risk_free_rate: Безрисковая ставка (annual)
        """
        self.portfolio = portfolio
        self.risk_free_rate = risk_free_rate
        
        logger.info("PerformanceAnalyzer инициализирован")
    
    def calculate_returns(self) -> pd.Series:
        """Рассчитать returns из equity curve"""
        equity_df = self.portfolio.get_equity_curve()
        if equity_df.empty or len(equity_df) < 2:
            return pd.Series(dtype=float)
        
        returns = equity_df['equity'].pct_change().dropna()
        return returns
    
    def calculate_sharpe_ratio(self, periods_per_year: int = 252) -> float:
        """
        Sharpe Ratio
        
        Args:
            periods_per_year: Количество периодов в год (252 для дней, 52 для недель)
        
        Returns:
            float: Sharpe Ratio
        """
        returns = self.calculate_returns()
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = np.sqrt(periods_per_year) * (excess_returns.mean() / excess_returns.std())
        return sharpe
    
    def calculate_sortino_ratio(self, periods_per_year: int = 252) -> float:
        """
        Sortino Ratio (учитывает только downside volatility)
        
        Args:
            periods_per_year: Количество периодов в год
        
        Returns:
            float: Sortino Ratio
        """
        returns = self.calculate_returns()
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        
        # Downside deviation (только отрицательные returns)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        downside_std = downside_returns.std()
        sortino = np.sqrt(periods_per_year) * (excess_returns.mean() / downside_std)
        return sortino
    
    def calculate_max_drawdown(self) -> Dict[str, float]:
        """
        Maximum Drawdown и связанные метрики
        
        Returns:
            dict: {'max_drawdown': float, 'max_drawdown_duration': int, 'current_drawdown': float}
        """
        equity_df = self.portfolio.get_equity_curve()
        if equity_df.empty:
            return {'max_drawdown': 0.0, 'max_drawdown_duration': 0, 'current_drawdown': 0.0}
        
        equity = equity_df['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        
        max_dd = drawdown.min()
        current_dd = drawdown.iloc[-1] if len(drawdown) > 0 else 0.0
        
        # Длительность максимального drawdown
        is_dd = drawdown < 0
        dd_periods = []
        current_period = 0
        
        for val in is_dd:
            if val:
                current_period += 1
            else:
                if current_period > 0:
                    dd_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            dd_periods.append(current_period)
        
        max_dd_duration = max(dd_periods) if dd_periods else 0
        
        return {
            'max_drawdown': abs(max_dd),
            'max_drawdown_duration': max_dd_duration,
            'current_drawdown': abs(current_dd)
        }
    
    def calculate_calmar_ratio(self) -> float:
        """
        Calmar Ratio = Annual Return / Max Drawdown
        
        Returns:
            float: Calmar Ratio
        """
        annual_return = self.calculate_annual_return()
        max_dd = self.calculate_max_drawdown()['max_drawdown']
        
        if max_dd == 0:
            return 0.0
        
        calmar = annual_return / max_dd
        return calmar
    
    def calculate_annual_return(self) -> float:
        """
        Аннуализированная доходность
        
        Returns:
            float: Annual return (в процентах)
        """
        equity_df = self.portfolio.get_equity_curve()
        if equity_df.empty or len(equity_df) < 2:
            return 0.0
        
        start_equity = equity_df['equity'].iloc[0]
        end_equity = equity_df['equity'].iloc[-1]
        
        if start_equity == 0:
            return 0.0
        
        total_return = (end_equity - start_equity) / start_equity
        
        # Аннуализация
        days = (equity_df.index[-1] - equity_df.index[0]).days
        if days == 0:
            return 0.0
        
        years = days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1
        
        return annual_return * 100
    
    def calculate_win_rate(self) -> float:
        """
        Win Rate (процент прибыльных сделок)
        
        Returns:
            float: Win rate (0 to 1)
        """
        trades_df = self.portfolio.get_trades_df()
        if trades_df.empty:
            return 0.0
        
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        total_trades = len(trades_df)
        
        return winning_trades / total_trades if total_trades > 0 else 0.0
    
    def calculate_profit_factor(self) -> float:
        """
        Profit Factor = Gross Profit / Gross Loss
        
        Returns:
            float: Profit factor
        """
        trades_df = self.portfolio.get_trades_df()
        if trades_df.empty:
            return 0.0
        
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        
        if gross_loss == 0:
            return 0.0 if gross_profit == 0 else float('inf')
        
        return gross_profit / gross_loss
    
    def calculate_average_trade(self) -> Dict[str, float]:
        """
        Средняя сделка (общая, выигрышная, проигрышная)
        
        Returns:
            dict: {'avg_trade': float, 'avg_win': float, 'avg_loss': float}
        """
        trades_df = self.portfolio.get_trades_df()
        if trades_df.empty:
            return {'avg_trade': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0}
        
        avg_trade = trades_df['pnl'].mean()
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0.0
        avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0.0
        
        return {
            'avg_trade': avg_trade,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
    
    def calculate_recovery_factor(self) -> float:
        """
        Recovery Factor = Net Profit / Max Drawdown
        
        Returns:
            float: Recovery factor
        """
        net_profit = self.portfolio.equity - self.portfolio.initial_capital
        max_dd = self.calculate_max_drawdown()['max_drawdown']
        
        if max_dd == 0:
            return 0.0
        
        # Max DD в абсолютных единицах
        max_dd_absolute = max_dd * self.portfolio.peak_equity
        
        if max_dd_absolute == 0:
            return 0.0
        
        return net_profit / max_dd_absolute
    
    def calculate_ulcer_index(self) -> float:
        """
        Ulcer Index (мера downside risk)
        
        Returns:
            float: Ulcer Index
        """
        equity_df = self.portfolio.get_equity_curve()
        if equity_df.empty or len(equity_df) < 2:
            return 0.0
        
        equity = equity_df['equity']
        running_max = equity.expanding().max()
        drawdown_pct = ((equity - running_max) / running_max) * 100
        
        # Ulcer Index = sqrt(mean(drawdown^2))
        ulcer = np.sqrt((drawdown_pct ** 2).mean())
        return ulcer
    
    def calculate_monthly_returns(self) -> pd.DataFrame:
        """
        Доходность по месяцам
        
        Returns:
            DataFrame: Месячная доходность
        """
        equity_df = self.portfolio.get_equity_curve()
        if equity_df.empty:
            return pd.DataFrame()
        
        # Resample to monthly
        monthly_equity = equity_df['equity'].resample('M').last()
        monthly_returns = monthly_equity.pct_change() * 100
        
        # Создаем таблицу year x month
        result = pd.DataFrame()
        result['year'] = monthly_returns.index.year
        result['month'] = monthly_returns.index.month
        result['return'] = monthly_returns.values
        
        pivot = result.pivot(index='year', columns='month', values='return')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        return pivot
    
    def calculate_trade_statistics(self) -> Dict[str, any]:
        """
        Подробная статистика по сделкам
        
        Returns:
            dict: Статистика сделок
        """
        trades_df = self.portfolio.get_trades_df()
        
        if trades_df.empty:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_holding_period': 0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0
            }
        
        winning = trades_df[trades_df['pnl'] > 0]
        losing = trades_df[trades_df['pnl'] < 0]
        
        # Consecutive wins/losses
        is_win = (trades_df['pnl'] > 0).astype(int)
        consecutive_wins = []
        consecutive_losses = []
        current_streak = 0
        last_val = None
        
        for val in is_win:
            if val == 1:  # Win
                if last_val == 1:
                    current_streak += 1
                else:
                    if current_streak > 0 and last_val == 0:
                        consecutive_losses.append(current_streak)
                    current_streak = 1
            else:  # Loss
                if last_val == 0:
                    current_streak += 1
                else:
                    if current_streak > 0 and last_val == 1:
                        consecutive_wins.append(current_streak)
                    current_streak = 1
            last_val = val
        
        # Add final streak
        if last_val == 1:
            consecutive_wins.append(current_streak)
        elif last_val == 0:
            consecutive_losses.append(current_streak)
        
        return {
            'total_trades': len(trades_df),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': self.calculate_win_rate(),
            'avg_holding_period': trades_df['holding_period'].mean() if 'holding_period' in trades_df.columns else 0,
            'max_consecutive_wins': max(consecutive_wins) if consecutive_wins else 0,
            'max_consecutive_losses': max(consecutive_losses) if consecutive_losses else 0,
            'largest_win': winning['pnl'].max() if not winning.empty else 0.0,
            'largest_loss': losing['pnl'].min() if not losing.empty else 0.0
        }
    
    def generate_report(self) -> Dict[str, any]:
        """
        Генерация полного отчета
        
        Returns:
            dict: Полный отчет по всем метрикам
        """
        portfolio_summary = self.portfolio.get_summary()
        dd_metrics = self.calculate_max_drawdown()
        avg_trade = self.calculate_average_trade()
        trade_stats = self.calculate_trade_statistics()
        
        report = {
            # Portfolio metrics
            'initial_capital': portfolio_summary['initial_capital'],
            'final_equity': portfolio_summary['current_equity'],
            'net_profit': portfolio_summary['total_pnl'],
            'total_return_pct': portfolio_summary['total_return_pct'],
            
            # Risk-adjusted metrics
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'sortino_ratio': self.calculate_sortino_ratio(),
            'calmar_ratio': self.calculate_calmar_ratio(),
            
            # Drawdown metrics
            'max_drawdown': dd_metrics['max_drawdown'],
            'max_drawdown_duration': dd_metrics['max_drawdown_duration'],
            'current_drawdown': dd_metrics['current_drawdown'],
            'ulcer_index': self.calculate_ulcer_index(),
            
            # Trade metrics
            'total_trades': trade_stats['total_trades'],
            'winning_trades': trade_stats['winning_trades'],
            'losing_trades': trade_stats['losing_trades'],
            'win_rate': trade_stats['win_rate'],
            'profit_factor': self.calculate_profit_factor(),
            
            # Average trade
            'avg_trade': avg_trade['avg_trade'],
            'avg_win': avg_trade['avg_win'],
            'avg_loss': avg_trade['avg_loss'],
            
            # Other
            'recovery_factor': self.calculate_recovery_factor(),
            'annual_return': self.calculate_annual_return(),
            'max_consecutive_wins': trade_stats['max_consecutive_wins'],
            'max_consecutive_losses': trade_stats['max_consecutive_losses'],
            
            # Costs
            'total_commission': portfolio_summary['total_commission'],
            'total_slippage': portfolio_summary['total_slippage']
        }
        
        return report
    
    def print_report(self):
        """Вывести отчет в консоль"""
        report = self.generate_report()
        
        print("\n" + "=" * 70)
        print(f"  BACKTEST PERFORMANCE REPORT: {self.portfolio.name}")
        print("=" * 70)
        
        print("\n📊 PORTFOLIO METRICS:")
        print(f"  Initial Capital:       ${report['initial_capital']:>15,.2f}")
        print(f"  Final Equity:          ${report['final_equity']:>15,.2f}")
        print(f"  Net Profit:            ${report['net_profit']:>15,.2f}")
        print(f"  Total Return:          {report['total_return_pct']:>15.2f}%")
        print(f"  Annual Return:         {report['annual_return']:>15.2f}%")
        
        print("\n📈 RISK-ADJUSTED RETURNS:")
        print(f"  Sharpe Ratio:          {report['sharpe_ratio']:>15.3f}")
        print(f"  Sortino Ratio:         {report['sortino_ratio']:>15.3f}")
        print(f"  Calmar Ratio:          {report['calmar_ratio']:>15.3f}")
        
        print("\n📉 DRAWDOWN METRICS:")
        print(f"  Max Drawdown:          {report['max_drawdown']*100:>15.2f}%")
        print(f"  Max DD Duration:       {report['max_drawdown_duration']:>15} periods")
        print(f"  Current Drawdown:      {report['current_drawdown']*100:>15.2f}%")
        print(f"  Ulcer Index:           {report['ulcer_index']:>15.2f}")
        print(f"  Recovery Factor:       {report['recovery_factor']:>15.2f}")
        
        print("\n💰 TRADE STATISTICS:")
        print(f"  Total Trades:          {report['total_trades']:>15}")
        print(f"  Winning Trades:        {report['winning_trades']:>15}")
        print(f"  Losing Trades:         {report['losing_trades']:>15}")
        print(f"  Win Rate:              {report['win_rate']*100:>15.2f}%")
        print(f"  Profit Factor:         {report['profit_factor']:>15.2f}")
        
        print("\n💵 AVERAGE TRADE:")
        print(f"  Average Trade:         ${report['avg_trade']:>15,.2f}")
        print(f"  Average Win:           ${report['avg_win']:>15,.2f}")
        print(f"  Average Loss:          ${report['avg_loss']:>15,.2f}")
        print(f"  Max Consecutive Wins:  {report['max_consecutive_wins']:>15}")
        print(f"  Max Consecutive Losses:{report['max_consecutive_losses']:>15}")
        
        print("\n💸 COSTS:")
        print(f"  Total Commission:      ${report['total_commission']:>15,.2f}")
        print(f"  Total Slippage:        ${report['total_slippage']:>15,.2f}")
        
        print("\n" + "=" * 70 + "\n")
