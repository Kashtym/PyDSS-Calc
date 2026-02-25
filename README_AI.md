# AI Context: DC Substation Auxiliary Power System Calculator (СВП ПС)

## Project Overview

Проєкт для розрахунку параметрів мереж постійного струму власних потреб підстанцій.

-   **Мета**: Розрахунок потокорозподілу, падіння напруги та струмів КЗ.
-   **Особливість**: Моделювання АБ (104 елементи) та перевірка селективності захисту (ВТХ).

## Технологічний Стек (Tech Stack)

-   **Core Engine**: OpenDSS (через бібліотеку `opendssdirect.py`).
-   **Backend/Logic**: Python 3.11+.
-   **UI Framework**: Streamlit (для візуалізації та введення даних).
-   **Data Storage**: `.ods` (LibreOffice Calc) для бази обладнання (кабелі, АБ, АВ).
-   **Math/Graphs**: NumPy, SciPy (інтерполяція ВТХ), Matplotlib/Plotly (графіки).

## Структура Проєкту (Project Structure)

/
├── app.py # Головний файл Streamlit (UI)
├── engine/ # Логіка взаємодії з OpenDSS
│ └── solver.py # Генерація .dss скриптів та запуск розрахунку
├── data/ # Бази даних обладнання
│ ├── library.ods # Головна база (Листи: Cables, Batteries, Breakers)
│ └── curves/ # CSV точки для ВТХ (Time-Current Curves)
├── utils/ # Допоміжні скрипти
│ └── approximation.py # Математична апроксимація кривих захисту
└── reports/ # Вихідні PDF звіти

## Technical Modeling Rules (Critical)

1. **DC System via OpenDSS**:
    - Simulate as AC system with Frequency = 0.001 Hz or 50 Hz.
    - Force `X=0` and `C=0` for all `Line` and `Vsource` objects to emulate pure DC.
2. **Battery (Storage/Source)**:
    - Model as `Vsource`.
    - Calculate total resistance: $R_{total} = (N_{cells} \times R_{cell\_mOhm} / 1000) + R_{jumpers}$.
3. **Cable Resistance Temperature Correction**:
    - $R_t = R_{20} \times [1 + 0.004 \times (T - 20)]$ for Copper.
4. **Selectivity Analysis**:
    - Use `scipy.interpolate.interp1d` in log-log scale for TCC (Time-Current Curves) mapping from CSV files.

## Одиниці виміру (Units)

-   Опір: Ом/км (для ліній), мОм (для комірок АБ).
-   Напруга: В або кВ (для OpenDSS), В (для UI).
-   Струм: А або кА.
-   Час: секунди.
