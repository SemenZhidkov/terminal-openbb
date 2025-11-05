"""
ML Backtester
Специализированный бэктестер для ML-моделей
"""

import logging
from typing import Dict, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np

from .backtester import Backtester
from .base import PositionSide, OrderType

logger = logging.getLogger(__name__)


class MLBacktester(Backtester):
    """
    Бэктестер для ML-стратегий
    
    Features:
    - Автоматический predict на каждом баре
    - Управление threshold для сигналов
    - Проверка доступности фич
    - Walk-forward совместимость
    - Метрики специфичные для ML
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        model: Any,  # sklearn-compatible model
        feature_columns: list,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        position_size: float = 1.0,
        prediction_threshold: float = 0.5,  # Threshold для сигналов
        name: str = "ML_Backtest"
    ):
        """
        Инициализация ML бэктестера
        
        Args:
            data: DataFrame с OHLCV и фичами
            model: Обученная ML модель (должна иметь метод predict или predict_proba)
            feature_columns: Список колонок-фич для предсказания
            initial_capital: Начальный капитал
            commission_rate: Комиссия
            slippage_rate: Проскальзывание
            position_size: Размер позиции
            prediction_threshold: Порог для генерации сигналов
            name: Название бэктеста
        """
        # ML-специфичные параметры
        self.model = model
        self.feature_columns = feature_columns
        self.prediction_threshold = prediction_threshold
        
        # Проверка наличия фич в данных
        missing_features = [f for f in feature_columns if f not in data.columns]
        if missing_features:
            raise ValueError(f"Отсутствуют фичи в данных: {missing_features}")
        
        # Инициализация базового бэктестера с ML-стратегией
        super().__init__(
            data=data,
            strategy=self._ml_strategy,
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            position_size=position_size,
            name=name
        )
        
        # Дополнительное хранилище для предсказаний
        self.predictions: list = []
        
        logger.info(f"ML Backtester '{name}' инициализирован с {len(feature_columns)} фичами")
    
    def _ml_strategy(self, bar: pd.Series, portfolio, context: dict) -> int:
        """
        ML стратегия: делает предсказание и генерирует сигнал
        
        Returns:
            1: BUY
            -1: SELL
            0: HOLD
        """
        symbol = bar.get('symbol', 'UNKNOWN')
        
        # Извлекаем фичи
        try:
            features = bar[self.feature_columns].values.reshape(1, -1)
            # Конвертируем в float для проверки NaN
            features = features.astype(float)
        except Exception as e:
            logger.warning(f"Ошибка извлечения фич: {e}")
            return 0
        
        # Проверка на NaN в фичах
        try:
            if np.any(np.isnan(features)):
                logger.debug(f"NaN в фичах на {bar.name}, пропуск")
                return 0
        except Exception:
            # Если не удается проверить NaN, пропускаем
            pass
        
        # Делаем предсказание
        try:
            if hasattr(self.model, 'predict_proba'):
                # Для классификаторов с вероятностями
                proba = self.model.predict_proba(features)[0]
                # Предполагаем binary classification: [prob_class_0, prob_class_1]
                prediction = proba[1] if len(proba) > 1 else proba[0]
            elif hasattr(self.model, 'predict'):
                # Для регрессоров или классификаторов без predict_proba
                prediction = self.model.predict(features)[0]
            else:
                logger.error("Модель не имеет методов predict/predict_proba")
                return 0
        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            return 0
        
        # Сохраняем предсказание
        self.predictions.append({
            'timestamp': bar.name,
            'prediction': prediction,
            'price': bar.get('close', 0)
        })
        
        # Генерируем сигнал на основе threshold
        has_position = portfolio.has_position(symbol)
        
        if prediction > self.prediction_threshold:
            # Модель предсказывает рост - покупаем
            if not has_position:
                return 1  # BUY
        elif prediction < (1 - self.prediction_threshold):
            # Модель предсказывает падение - продаем
            if has_position:
                return -1  # SELL
        
        return 0  # HOLD
    
    def run(self, verbose: bool = True) -> Dict:
        """
        Запуск ML бэктеста
        
        Args:
            verbose: Показывать progress bar
        
        Returns:
            dict: Результаты с метриками и предсказаниями
        """
        # Очистка предсказаний
        self.predictions.clear()
        
        # Запуск базового бэктеста
        results = super().run(verbose=verbose)
        
        # Добавляем предсказания в результаты
        if self.predictions:
            results['predictions'] = pd.DataFrame(self.predictions).set_index('timestamp')
        
        # ML-специфичные метрики
        results['ml_metrics'] = self._calculate_ml_metrics()
        
        return results
    
    def _calculate_ml_metrics(self) -> Dict:
        """
        Расчет ML-специфичных метрик
        
        Returns:
            dict: Метрики предсказаний
        """
        if not self.predictions:
            return {}
        
        predictions_df = pd.DataFrame(self.predictions)
        
        metrics = {
            'total_predictions': len(predictions_df),
            'avg_prediction': predictions_df['prediction'].mean(),
            'prediction_std': predictions_df['prediction'].std(),
            'prediction_threshold': self.prediction_threshold,
            'feature_count': len(self.feature_columns)
        }
        
        # Если есть сделки, добавляем корреляцию предсказаний с P&L
        trades_df = self.portfolio.get_trades_df()
        if not trades_df.empty and 'predictions' in self.get_results():
            pred_df = self.get_results()['predictions']
            
            # Merge predictions with trades
            trades_with_pred = trades_df.merge(
                pred_df, 
                left_on='entry_time', 
                right_index=True, 
                how='left'
            )
            
            if 'prediction' in trades_with_pred.columns:
                corr = trades_with_pred['prediction'].corr(trades_with_pred['pnl'])
                metrics['prediction_pnl_correlation'] = corr
        
        return metrics
    
    def print_ml_summary(self):
        """Вывести сводку по ML-метрикам"""
        results = self.get_results()
        ml_metrics = results.get('ml_metrics', {})
        
        print("\n" + "=" * 70)
        print("  ML MODEL METRICS")
        print("=" * 70)
        
        print(f"\n🤖 MODEL INFO:")
        print(f"  Model Type:            {type(self.model).__name__}")
        print(f"  Feature Count:         {ml_metrics.get('feature_count', 0)}")
        print(f"  Prediction Threshold:  {ml_metrics.get('prediction_threshold', 0):.3f}")
        
        print(f"\n📊 PREDICTION STATISTICS:")
        print(f"  Total Predictions:     {ml_metrics.get('total_predictions', 0)}")
        print(f"  Avg Prediction:        {ml_metrics.get('avg_prediction', 0):.3f}")
        print(f"  Prediction StdDev:     {ml_metrics.get('prediction_std', 0):.3f}")
        
        if 'prediction_pnl_correlation' in ml_metrics:
            corr = ml_metrics['prediction_pnl_correlation']
            print(f"  Prediction-PnL Corr:   {corr:.3f}")
        
        print("\n" + "=" * 70 + "\n")
    
    def plot_predictions(self, figsize=(15, 8)):
        """
        Визуализация предсказаний модели
        
        Args:
            figsize: Размер фигуры
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.predictions:
                logger.warning("Нет предсказаний для визуализации")
                return
            
            pred_df = pd.DataFrame(self.predictions).set_index('timestamp')
            equity_df = self.portfolio.get_equity_curve()
            
            fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
            
            # 1. Цена
            ax1 = axes[0]
            ax1.plot(pred_df.index, pred_df['price'], label='Price', linewidth=2)
            ax1.set_ylabel('Price ($)')
            ax1.set_title(f'{self.name}: Price Chart', fontsize=14, fontweight='bold')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            
            # 2. Предсказания
            ax2 = axes[1]
            ax2.plot(pred_df.index, pred_df['prediction'], label='ML Prediction', 
                    linewidth=2, color='orange')
            ax2.axhline(self.prediction_threshold, color='green', linestyle='--', 
                       alpha=0.5, label='Buy Threshold')
            ax2.axhline(1 - self.prediction_threshold, color='red', linestyle='--', 
                       alpha=0.5, label='Sell Threshold')
            ax2.set_ylabel('Prediction')
            ax2.set_title('ML Model Predictions', fontsize=12)
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
            
            # 3. Equity
            ax3 = axes[2]
            if not equity_df.empty:
                ax3.plot(equity_df.index, equity_df['equity'], 
                        label='Equity', linewidth=2, color='blue')
                ax3.axhline(self.portfolio.initial_capital, color='gray', 
                           linestyle='--', alpha=0.5)
            ax3.set_ylabel('Equity ($)')
            ax3.set_xlabel('Date')
            ax3.set_title('Equity Curve', fontsize=12)
            ax3.legend(loc='best')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib не установлен")


def create_dummy_ml_model(feature_count: int = 10):
    """
    Создать dummy ML модель для тестирования
    
    Args:
        feature_count: Количество фич
    
    Returns:
        Dummy модель с методом predict
    """
    class DummyModel:
        def __init__(self, n_features):
            self.n_features = n_features
            # Случайные веса
            self.weights = np.random.randn(n_features)
        
        def predict(self, X):
            # Простая линейная комбинация + sigmoid
            z = np.dot(X, self.weights)
            return 1 / (1 + np.exp(-z))
        
        def predict_proba(self, X):
            pred = self.predict(X)
            return np.column_stack([1 - pred, pred])
    
    return DummyModel(feature_count)
