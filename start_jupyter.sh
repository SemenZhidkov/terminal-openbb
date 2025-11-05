#!/bin/bash
# Скрипт для запуска Jupyter Lab из корня проекта
# Это гарантирует, что все относительные пути будут работать корректно

cd "$(dirname "$0")"
echo "📂 Запуск Jupyter Lab из директории: $(pwd)"
echo "🔗 Jupyter Lab откроется в браузере через несколько секунд..."
echo ""
jupyter lab notebooks/exploratory/02_feature_engineering_demo.ipynb
