#!/usr/bin/env python3
"""
Скрипт ежедневного обновления финансовых данных
"""

import sys
import os
import logging
from datetime import datetime
import pandas as pd

# Добавляем путь к src для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_manager import DataManager
from src.data.data_validator import DataValidator

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_update.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class DailyDataUpdater:
    """Класс для ежедневного обновления данных"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.validator = DataValidator(self.data_manager.config)
        self.results = {
            'timestamp': datetime.now(),
            'successful': [],
            'failed': [],
            'validation_results': {}
        }
    
    def run_update(self):
        """Запуск процесса обновления"""
        logger.info("Запуск ежедневного обновления данных...")
        
        try:
            # Пакетная загрузка всех данных
            batch_results = self.data_manager.batch_download()
            
            # Валидация загруженных данных
            self._validate_all_data(batch_results)
            
            # Создание отчета
            self._generate_report()
            
            # Очистка старых данных
            self._cleanup_old_data()
            
            logger.info("Ежедневное обновление завершено успешно")
            
        except Exception as e:
            logger.error(f"Критическая ошибка при обновлении данных: {e}")
            raise
    
    def _validate_all_data(self, batch_results: dict):
        """Валидация всех загруженных данных"""
        for asset_type, symbols in batch_results.items():
            for symbol, timeframes in symbols.items():
                for timeframe, data in timeframes.items():
                    if data is not None and not data.empty:
                        validation_result = self.validator.validate_dataset(data, symbol)
                        
                        self.results['validation_results'][f"{symbol}_{timeframe}"] = validation_result
                        
                        if validation_result['is_valid']:
                            self.results['successful'].append(f"{symbol}_{timeframe}")
                            logger.info(f"✓ {symbol} ({timeframe}): Валидация пройдена")
                        else:
                            self.results['failed'].append(f"{symbol}_{timeframe}")
                            logger.warning(f"✗ {symbol} ({timeframe}): Ошибки валидации")
                            
                            # Сохраняем детали ошибок
                            for error in validation_result['errors']:
                                logger.error(f"  - {error}")
    
    def _generate_report(self):
        """Генерация отчета об обновлении"""
        report = {
            'update_timestamp': self.results['timestamp'].isoformat(),
            'summary': {
                'total_attempted': len(self.results['successful']) + len(self.results['failed']),
                'successful': len(self.results['successful']),
                'failed': len(self.results['failed']),
                'success_rate': len(self.results['successful']) / (len(self.results['successful']) + len(self.results['failed'])) * 100
            },
            'cache_info': self.data_manager.get_cache_info(),
            'failed_assets': self.results['failed'],
            'validation_summary': {}
        }
        
        # Статистика валидации
        validation_stats = {
            'total_validations': len(self.results['validation_results']),
            'completely_valid': 0,
            'with_warnings': 0,
            'with_errors': 0
        }
        
        for result in self.results['validation_results'].values():
            if result['is_valid'] and not result['warnings']:
                validation_stats['completely_valid'] += 1
            elif result['is_valid'] and result['warnings']:
                validation_stats['with_warnings'] += 1
            else:
                validation_stats['with_errors'] += 1
        
        report['validation_summary'] = validation_stats
        
        # Сохранение отчета
        report_file = f"data/backup/update_report_{self.results['timestamp'].strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs('data/backup', exist_ok=True)
        
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Отчет сохранен: {report_file}")
        
        # Вывод краткого отчета в консоль
        self._print_summary(report)
    
    def _print_summary(self, report: dict):
        """Вывод краткого отчета в консоль"""
        summary = report['summary']
        val_summary = report['validation_summary']
        
        print("\n" + "="*50)
        print("ОТЧЕТ ОБ ОБНОВЛЕНИИ ДАННЫХ")
        print("="*50)
        print(f"Время обновления: {report['update_timestamp']}")
        print(f"Всего активов: {summary['total_attempted']}")
        print(f"Успешно: {summary['successful']}")
        print(f"С ошибками: {summary['failed']}")
        print(f"Успешность: {summary['success_rate']:.1f}%")
        print(f"Размер кэша: {report['cache_info']['total_size_gb']:.2f} GB")
        print(f"Файлов в кэше: {report['cache_info']['total_files']}")
        print("\nСтатистика валидации:")
        print(f"  Полностью валидны: {val_summary['completely_valid']}")
        print(f"  С предупреждениями: {val_summary['with_warnings']}")
        print(f"  С ошибками: {val_summary['with_errors']}")
        print("="*50)
    
    def _cleanup_old_data(self, days_to_keep: int = 30):
        """Очистка старых backup файлов"""
        try:
            backup_dir = Path('data/backup')
            if not backup_dir.exists():
                return
            
            cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
            
            for file in backup_dir.glob('*.json'):
                if file.stat().st_mtime < cutoff_time:
                    file.unlink()
                    logger.info(f"Удален старый backup: {file}")
                    
        except Exception as e:
            logger.warning(f"Ошибка при очистке старых данных: {e}")

def main():
    """Основная функция"""
    try:
        updater = DailyDataUpdater()
        updater.run_update()
        
    except Exception as e:
        logger.error(f"Скрипт завершился с ошибкой: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()