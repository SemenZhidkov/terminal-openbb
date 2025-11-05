"""
Walk-Forward Analysis
Система для тестирования устойчивости стратегий
"""

import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm

from .backtester import Backtester
from .ml_backtester import MLBacktester

logger = logging.getLogger(__name__)


class WalkForwardAnalyzer:
    """
    Walk-Forward анализ для тестирования устойчивости стратегий
    
    Процесс:
    1. Разбить данные на train/test windows
    2. Обучить на train window
    3. Тестировать на следующем test window
    4. Сдвинуть окна и повторить
    5. Агрегировать результаты
    
    Features:
    - Rolling windows
    - Anchored windows
    - Custom optimization functions
    - Stability metrics
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        train_period: int,  # Количество дней для обучения
        test_period: int,   # Количество дней для тестирования
        optimization_func: Optional[Callable] = None,  # Функция оптимизации параметров
        step_size: Optional[int] = None,  # Шаг сдвига окна (None = test_period)
        anchored: bool = False,  # True = anchored, False = rolling
        name: str = "WalkForward"
    ):
        """
        Инициализация Walk-Forward анализа
        
        Args:
            data: Полный датасет
            train_period: Размер обучающего окна (в днях)
            test_period: Размер тестового окна (в днях)
            optimization_func: Функция для оптимизации на train (опционально)
            step_size: Шаг сдвига окна (по умолчанию = test_period)
            anchored: Если True, train окно растет, если False - скользящее
            name: Название анализа
        """
        self.data = data.copy()
        self.train_period = train_period
        self.test_period = test_period
        self.optimization_func = optimization_func
        self.step_size = step_size if step_size else test_period
        self.anchored = anchored
        self.name = name
        
        # Результаты
        self.windows: List[Dict] = []
        self.results: Dict = {}
        
        logger.info(
            f"WalkForwardAnalyzer инициализирован: "
            f"train={train_period}d, test={test_period}d, "
            f"step={self.step_size}d, anchored={anchored}"
        )
    
    def _split_windows(self) -> List[Dict]:
        """
        Разбить данные на train/test windows
        
        Returns:
            List[dict]: Список окон с train/test индексами
        """
        windows = []
        
        total_days = len(self.data)
        start_idx = 0
        
        while start_idx + self.train_period + self.test_period <= total_days:
            if self.anchored:
                # Anchored: train окно от начала
                train_start = 0
                train_end = start_idx + self.train_period
            else:
                # Rolling: train окно скользит
                train_start = start_idx
                train_end = start_idx + self.train_period
            
            test_start = train_end
            test_end = test_start + self.test_period
            
            windows.append({
                'window_id': len(windows) + 1,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_data': self.data.iloc[train_start:train_end],
                'test_data': self.data.iloc[test_start:test_end]
            })
            
            # Сдвигаем окно
            start_idx += self.step_size
            
            # Если anchored и уже дошли до конца - выходим
            if self.anchored and test_end >= total_days:
                break
        
        logger.info(f"Создано {len(windows)} walk-forward windows")
        return windows
    
    def run(
        self,
        strategy_func: Callable,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        position_size: float = 1.0,
        verbose: bool = True
    ) -> Dict:
        """
        Запуск walk-forward анализа с простой стратегией
        
        Args:
            strategy_func: Функция стратегии
            initial_capital: Начальный капитал
            commission_rate: Комиссия
            slippage_rate: Проскальзывание
            position_size: Размер позиции
            verbose: Показывать progress
        
        Returns:
            dict: Агрегированные результаты
        """
        logger.info(f"Запуск walk-forward анализа '{self.name}'...")
        
        # Разбиваем на окна
        windows = self._split_windows()
        self.windows = windows
        
        # Итерируемся по окнам
        iterator = tqdm(windows, desc="Walk-Forward Windows") if verbose else windows
        
        for window in iterator:
            window_id = window['window_id']
            train_data = window['train_data']
            test_data = window['test_data']
            
            logger.info(
                f"Window {window_id}: "
                f"Train: {train_data.index[0]} to {train_data.index[-1]}, "
                f"Test: {test_data.index[0]} to {test_data.index[-1]}"
            )
            
            # Оптимизация параметров на train (если есть функция)
            optimized_params = {}
            if self.optimization_func:
                try:
                    optimized_params = self.optimization_func(train_data, window_id)
                    logger.info(f"Оптимизированные параметры: {optimized_params}")
                except Exception as e:
                    logger.error(f"Ошибка оптимизации: {e}")
            
            # Тестирование на test window
            backtester = Backtester(
                data=test_data,
                strategy=strategy_func,
                initial_capital=initial_capital,
                commission_rate=commission_rate,
                slippage_rate=slippage_rate,
                position_size=position_size,
                name=f"{self.name}_Window_{window_id}"
            )
            
            # Передаем оптимизированные параметры через context
            backtester.context['optimized_params'] = optimized_params
            
            # Запуск бэктеста
            results = backtester.run(verbose=False)
            
            # Сохраняем результаты окна
            window['results'] = results
            window['optimized_params'] = optimized_params
        
        # Агрегируем результаты
        self.results = self._aggregate_results()
        
        logger.info(f"Walk-forward анализ '{self.name}' завершен")
        return self.results
    
    def run_ml(
        self,
        model_class: Any,  # Класс ML модели (например, RandomForestClassifier)
        feature_columns: List[str],
        target_column: str,
        model_params: Optional[Dict] = None,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        position_size: float = 1.0,
        prediction_threshold: float = 0.5,
        verbose: bool = True
    ) -> Dict:
        """
        Запуск walk-forward анализа с ML моделью
        
        Args:
            model_class: Класс модели (например, sklearn.RandomForestClassifier)
            feature_columns: Список фич
            target_column: Целевая переменная
            model_params: Параметры модели
            initial_capital: Начальный капитал
            commission_rate: Комиссия
            slippage_rate: Проскальзывание
            position_size: Размер позиции
            prediction_threshold: Порог предсказания
            verbose: Показывать progress
        
        Returns:
            dict: Агрегированные результаты
        """
        logger.info(f"Запуск ML walk-forward анализа '{self.name}'...")
        
        model_params = model_params or {}
        
        # Разбиваем на окна
        windows = self._split_windows()
        self.windows = windows
        
        # Итерируемся по окнам
        iterator = tqdm(windows, desc="ML Walk-Forward Windows") if verbose else windows
        
        for window in iterator:
            window_id = window['window_id']
            train_data = window['train_data']
            test_data = window['test_data']
            
            logger.info(
                f"Window {window_id}: Training on {len(train_data)} samples, "
                f"Testing on {len(test_data)} samples"
            )
            
            # Подготовка данных для обучения
            X_train = train_data[feature_columns].dropna()
            y_train = train_data.loc[X_train.index, target_column]
            
            # Обучение модели
            try:
                model = model_class(**model_params)
                model.fit(X_train, y_train)
                
                # Метрики на train
                train_score = model.score(X_train, y_train) if hasattr(model, 'score') else None
                window['train_score'] = train_score
                
                logger.info(f"Модель обучена. Train score: {train_score:.4f}" if train_score else "Модель обучена")
            except Exception as e:
                logger.error(f"Ошибка обучения модели: {e}")
                continue
            
            # Тестирование на test window
            try:
                ml_backtester = MLBacktester(
                    data=test_data,
                    model=model,
                    feature_columns=feature_columns,
                    initial_capital=initial_capital,
                    commission_rate=commission_rate,
                    slippage_rate=slippage_rate,
                    position_size=position_size,
                    prediction_threshold=prediction_threshold,
                    name=f"{self.name}_ML_Window_{window_id}"
                )
                
                results = ml_backtester.run(verbose=False)
                
                # Сохраняем результаты
                window['results'] = results
                window['model'] = model
                
            except Exception as e:
                logger.error(f"Ошибка бэктеста ML: {e}")
                continue
        
        # Агрегируем результаты
        self.results = self._aggregate_results()
        
        logger.info(f"ML walk-forward анализ '{self.name}' завершен")
        return self.results
    
    def _aggregate_results(self) -> Dict:
        """
        Агрегировать результаты всех окон
        
        Returns:
            dict: Сводные результаты
        """
        if not self.windows:
            return {}
        
        # Собираем метрики по всем окнам
        window_metrics = []
        
        for window in self.windows:
            if 'results' not in window:
                continue
            
            results = window['results']
            metrics = {
                'window_id': window['window_id'],
                'total_return': results.get('total_return_pct', 0),
                'sharpe_ratio': results.get('sharpe_ratio', 0),
                'max_drawdown': results.get('max_drawdown', 0),
                'win_rate': results.get('win_rate', 0),
                'profit_factor': results.get('profit_factor', 0),
                'total_trades': results.get('total_trades', 0),
                'net_profit': results.get('net_profit', 0)
            }
            
            if 'train_score' in window:
                metrics['train_score'] = window['train_score']
            
            window_metrics.append(metrics)
        
        metrics_df = pd.DataFrame(window_metrics)
        
        if metrics_df.empty:
            return {'error': 'Нет результатов для агрегации'}
        
        # Агрегированные метрики
        aggregated = {
            'total_windows': len(self.windows),
            'successful_windows': len(window_metrics),
            
            # Средние метрики
            'avg_return': metrics_df['total_return'].mean(),
            'avg_sharpe': metrics_df['sharpe_ratio'].mean(),
            'avg_max_drawdown': metrics_df['max_drawdown'].mean(),
            'avg_win_rate': metrics_df['win_rate'].mean(),
            'avg_profit_factor': metrics_df['profit_factor'].mean(),
            
            # Стабильность
            'return_std': metrics_df['total_return'].std(),
            'sharpe_std': metrics_df['sharpe_ratio'].std(),
            'positive_windows': (metrics_df['total_return'] > 0).sum(),
            'negative_windows': (metrics_df['total_return'] < 0).sum(),
            
            # Суммарные метрики
            'total_trades': metrics_df['total_trades'].sum(),
            'total_net_profit': metrics_df['net_profit'].sum(),
            
            # Детальные данные
            'window_metrics': metrics_df,
            'windows': self.windows
        }
        
        # Stability Score (0-100)
        positive_pct = aggregated['positive_windows'] / aggregated['total_windows'] * 100
        sharpe_consistency = 100 - min(abs(aggregated['sharpe_std']) * 50, 100)
        stability_score = (positive_pct * 0.6) + (sharpe_consistency * 0.4)
        aggregated['stability_score'] = stability_score
        
        return aggregated
    
    def print_summary(self):
        """Вывести сводку walk-forward анализа"""
        if not self.results:
            logger.warning("Нет результатов для отображения")
            return
        
        r = self.results
        
        print("\n" + "=" * 70)
        print(f"  WALK-FORWARD ANALYSIS: {self.name}")
        print("=" * 70)
        
        print(f"\n⚙️ CONFIGURATION:")
        print(f"  Train Period:          {self.train_period} days")
        print(f"  Test Period:           {self.test_period} days")
        print(f"  Step Size:             {self.step_size} days")
        print(f"  Type:                  {'Anchored' if self.anchored else 'Rolling'}")
        print(f"  Total Windows:         {r['total_windows']}")
        print(f"  Successful Windows:    {r['successful_windows']}")
        
        print(f"\n📊 AGGREGATED PERFORMANCE:")
        print(f"  Avg Return:            {r['avg_return']:>15.2f}%")
        print(f"  Avg Sharpe Ratio:      {r['avg_sharpe']:>15.3f}")
        print(f"  Avg Max Drawdown:      {r['avg_max_drawdown']*100:>15.2f}%")
        print(f"  Avg Win Rate:          {r['avg_win_rate']*100:>15.2f}%")
        print(f"  Avg Profit Factor:     {r['avg_profit_factor']:>15.2f}")
        
        print(f"\n🎯 STABILITY METRICS:")
        print(f"  Positive Windows:      {r['positive_windows']}/{r['total_windows']}")
        print(f"  Negative Windows:      {r['negative_windows']}/{r['total_windows']}")
        print(f"  Return StdDev:         {r['return_std']:>15.2f}%")
        print(f"  Sharpe StdDev:         {r['sharpe_std']:>15.3f}")
        print(f"  Stability Score:       {r['stability_score']:>15.1f}/100")
        
        print(f"\n💰 CUMULATIVE:")
        print(f"  Total Trades:          {r['total_trades']}")
        print(f"  Total Net Profit:      ${r['total_net_profit']:>15,.2f}")
        
        print("\n" + "=" * 70 + "\n")
    
    def plot_results(self, figsize=(15, 12)):
        """
        Визуализация walk-forward результатов
        
        Args:
            figsize: Размер фигуры
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if not self.results or 'window_metrics' not in self.results:
                logger.warning("Нет данных для визуализации")
                return
            
            metrics_df = self.results['window_metrics']
            
            fig, axes = plt.subplots(3, 2, figsize=figsize)
            
            # 1. Returns по окнам
            ax1 = axes[0, 0]
            ax1.bar(metrics_df['window_id'], metrics_df['total_return'], 
                   color=['green' if x > 0 else 'red' for x in metrics_df['total_return']])
            ax1.axhline(0, color='black', linewidth=0.8)
            ax1.set_xlabel('Window ID')
            ax1.set_ylabel('Return (%)')
            ax1.set_title('Returns by Window')
            ax1.grid(True, alpha=0.3)
            
            # 2. Sharpe Ratio по окнам
            ax2 = axes[0, 1]
            ax2.plot(metrics_df['window_id'], metrics_df['sharpe_ratio'], 
                    marker='o', linewidth=2)
            ax2.axhline(metrics_df['sharpe_ratio'].mean(), color='red', 
                       linestyle='--', label='Average')
            ax2.set_xlabel('Window ID')
            ax2.set_ylabel('Sharpe Ratio')
            ax2.set_title('Sharpe Ratio by Window')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Max Drawdown
            ax3 = axes[1, 0]
            ax3.bar(metrics_df['window_id'], metrics_df['max_drawdown'] * 100)
            ax3.set_xlabel('Window ID')
            ax3.set_ylabel('Max Drawdown (%)')
            ax3.set_title('Max Drawdown by Window')
            ax3.grid(True, alpha=0.3)
            
            # 4. Win Rate
            ax4 = axes[1, 1]
            ax4.plot(metrics_df['window_id'], metrics_df['win_rate'] * 100, 
                    marker='s', linewidth=2, color='purple')
            ax4.axhline(50, color='gray', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Window ID')
            ax4.set_ylabel('Win Rate (%)')
            ax4.set_title('Win Rate by Window')
            ax4.grid(True, alpha=0.3)
            
            # 5. Profit Factor
            ax5 = axes[2, 0]
            ax5.bar(metrics_df['window_id'], metrics_df['profit_factor'], 
                   color='orange', alpha=0.7)
            ax5.axhline(1, color='red', linestyle='--', linewidth=2)
            ax5.set_xlabel('Window ID')
            ax5.set_ylabel('Profit Factor')
            ax5.set_title('Profit Factor by Window')
            ax5.grid(True, alpha=0.3)
            
            # 6. Distribution of Returns
            ax6 = axes[2, 1]
            ax6.hist(metrics_df['total_return'], bins=15, edgecolor='black', alpha=0.7)
            ax6.axvline(0, color='red', linestyle='--', linewidth=2)
            ax6.axvline(metrics_df['total_return'].mean(), color='green', 
                       linestyle='--', linewidth=2, label='Mean')
            ax6.set_xlabel('Return (%)')
            ax6.set_ylabel('Frequency')
            ax6.set_title('Distribution of Returns')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib не установлен")
