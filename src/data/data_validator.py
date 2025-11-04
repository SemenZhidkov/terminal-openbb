import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataValidator:
    """Класс для валидации финансовых данных"""
    
    def __init__(self, config: dict):
        self.config = config
        self.validation_rules = self._setup_validation_rules()
    
    def _setup_validation_rules(self) -> Dict:
        """Настройка правил валидации"""
        return {
            'min_data_points': self.config['data_settings']['validation']['min_data_points'],
            'max_null_percentage': self.config['data_settings']['validation']['max_null_percentage'],
            'price_variance_threshold': self.config['data_settings']['validation']['price_variance_threshold'],
            'required_columns': ['open', 'high', 'low', 'close', 'volume'],
            'price_consistency': {
                'high_low_check': True,
                'open_close_range': True
            }
        }
    
    def validate_dataset(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Полная валидация набора данных"""
        validation_results = {
            'symbol': symbol,
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'metrics': {},
            'passed_tests': 0,
            'total_tests': 0
        }
        
        tests = [
            self._check_data_size,
            self._check_required_columns,
            self._check_null_values,
            self._check_price_consistency,
            self._check_volume_data,
            self._check_date_integrity,
            self._check_price_variance
        ]
        
        for test in tests:
            try:
                test_result = test(data)
                validation_results['total_tests'] += 1
                
                if test_result['status'] == 'error':
                    validation_results['errors'].append(test_result['message'])
                    validation_results['is_valid'] = False
                elif test_result['status'] == 'warning':
                    validation_results['warnings'].append(test_result['message'])
                else:
                    validation_results['passed_tests'] += 1
                
                # Сохраняем метрики
                if 'metrics' in test_result:
                    validation_results['metrics'].update(test_result['metrics'])
                    
            except Exception as e:
                logger.error(f"Ошибка при выполнении теста {test.__name__}: {e}")
                validation_results['errors'].append(f"Тест {test.__name__} завершился ошибкой: {e}")
                validation_results['is_valid'] = False
        
        return validation_results
    
    def _check_data_size(self, data: pd.DataFrame) -> Dict:
        """Проверка минимального размера данных"""
        min_points = self.validation_rules['min_data_points']
        result = {'status': 'passed', 'message': f"Размер данных достаточен: {len(data)} записей"}
        
        if len(data) < min_points:
            result = {
                'status': 'error', 
                'message': f"Недостаточно данных: {len(data)} записей (минимум {min_points})"
            }
        
        result['metrics'] = {'data_points': len(data)}
        return result
    
    def _check_required_columns(self, data: pd.DataFrame) -> Dict:
        """Проверка наличия обязательных колонок"""
        missing_columns = [
            col for col in self.validation_rules['required_columns'] 
            if col not in data.columns
        ]
        
        if missing_columns:
            return {
                'status': 'error',
                'message': f"Отсутствуют обязательные колонки: {missing_columns}"
            }
        
        return {'status': 'passed', 'message': "Все обязательные колонки присутствуют"}
    
    def _check_null_values(self, data: pd.DataFrame) -> Dict:
        """Проверка пропущенных значений"""
        max_null_pct = self.validation_rules['max_null_percentage']
        null_stats = {}
        
        for column in self.validation_rules['required_columns']:
            if column in data.columns:
                null_count = data[column].isnull().sum()
                null_percentage = (null_count / len(data)) * 100
                null_stats[column] = null_percentage
                
                if null_percentage > max_null_pct:
                    return {
                        'status': 'error',
                        'message': f"Слишком много пропусков в {column}: {null_percentage:.2f}%"
                    }
        
        result = {'status': 'passed', 'message': "Проверка пропусков пройдена"}
        result['metrics'] = {'null_percentages': null_stats}
        return result
    
    def _check_price_consistency(self, data: pd.DataFrame) -> Dict:
        """Проверка консистентности ценовых данных"""
        errors = []
        
        # Проверка: high >= low
        if 'high' in data.columns and 'low' in data.columns:
            invalid_high_low = data[data['high'] < data['low']]
            if len(invalid_high_low) > 0:
                errors.append(f"Найдено {len(invalid_high_low)} записей где high < low")
        
        # Проверка: open и close в диапазоне high-low
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            invalid_open = data[
                (data['open'] > data['high']) | (data['open'] < data['low'])
            ]
            invalid_close = data[
                (data['close'] > data['high']) | (data['close'] < data['low'])
            ]
            
            if len(invalid_open) > 0:
                errors.append(f"Найдено {len(invalid_open)} записей с open вне диапазона high-low")
            if len(invalid_close) > 0:
                errors.append(f"Найдено {len(invalid_close)} записей с close вне диапазона high-low")
        
        if errors:
            return {'status': 'error', 'message': "; ".join(errors)}
        
        return {'status': 'passed', 'message': "Проверка консистентности цен пройдена"}
    
    def _check_volume_data(self, data: pd.DataFrame) -> Dict:
        """Проверка данных объема"""
        if 'volume' not in data.columns:
            return {'status': 'warning', 'message': "Колонка volume отсутствует"}
        
        zero_volume = data[data['volume'] == 0]
        negative_volume = data[data['volume'] < 0]
        
        warnings = []
        if len(zero_volume) > 0:
            warnings.append(f"Найдено {len(zero_volume)} записей с нулевым объемом")
        if len(negative_volume) > 0:
            warnings.append(f"Найдено {len(negative_volume)} записей с отрицательным объемом")
        
        if warnings:
            return {'status': 'warning', 'message': "; ".join(warnings)}
        
        return {'status': 'passed', 'message': "Проверка объема пройдена"}
    
    def _check_date_integrity(self, data: pd.DataFrame) -> Dict:
        """Проверка целостности временных рядов"""
        if data.index.name != 'date' and 'date' not in data.columns:
            return {'status': 'warning', 'message': "Не удалось проверить целостность дат"}
        
        # Используем индекс если это дата, иначе колонку date
        if data.index.name == 'date':
            dates = data.index
        else:
            dates = data['date']
        
        # Проверка на дубликаты дат
        duplicate_dates = dates.duplicated().sum()
        if duplicate_dates > 0:
            return {
                'status': 'warning', 
                'message': f"Найдено {duplicate_dates} дубликатов дат"
            }
        
        # Проверка на пропуски в датах (только для дневных данных)
        if len(dates) > 1:
            date_diff = pd.Series(dates).diff().dt.days
            unusual_gaps = date_diff[date_diff > 7].count()  # Пропуски больше недели
            
            if unusual_gaps > 0:
                return {
                    'status': 'warning',
                    'message': f"Найдено {unusual_gaps} необычно больших пропусков в датах"
                }
        
        return {'status': 'passed', 'message': "Проверка целостности дат пройдена"}
    
    def _check_price_variance(self, data: pd.DataFrame) -> Dict:
        """Проверка дисперсии цен"""
        threshold = self.validation_rules['price_variance_threshold']
        
        if 'close' not in data.columns:
            return {'status': 'warning', 'message': "Нет данных close для проверки дисперсии"}
        
        price_changes = data['close'].pct_change().dropna()
        
        if len(price_changes) == 0:
            return {'status': 'warning', 'message': "Недостаточно данных для расчета дисперсии"}
        
        # Проверяем на аномально низкую волатильность
        if price_changes.std() < threshold:
            return {
                'status': 'warning',
                'message': f"Обнаружена аномально низкая волатильность: {price_changes.std():.6f}"
            }
        
        result = {'status': 'passed', 'message': "Проверка дисперсии цен пройдена"}
        result['metrics'] = {'price_volatility': price_changes.std()}
        return result