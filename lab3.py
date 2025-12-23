import random
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import time


# Класс для ведения аналитики

class TunnelAnalytics:
    def __init__(self):
        self.history = []
        self.actions_taken = 0

    def log(self, vehicle_count, air_quality, fan_low, fan_high):
        self.history.append({
            "vehicle_count": vehicle_count,
            "air_quality": air_quality,
            "fan_low": fan_low,
            "fan_high": fan_high
        })
        self.actions_taken += 1

    def report(self):
        print("\n===== ОТЧЕТ СИСТЕМЫ УПРАВЛЕНИЯ ТУННЕЛЕМ =====")
        print(f"Всего управляющих воздействий: {self.actions_taken}")


# Онтология туннеля

def create_tunnel_ontology():
    return {
        "vehicle_range": (0, 100),
        "air_quality_range": (0, 100),  # 0 - очень загрязненный, 100 - чистый
        "optimal_air_quality": 80
    }


# Настройка нечеткой системы

def setup_fuzzy_systems(tunnel):
    # Вход: количество машин
    vehicle_count = ctrl.Antecedent(np.arange(0, 101, 1), 'vehicle_count')
    # Выход: скорость вентилятора
    fan_low = ctrl.Consequent(np.arange(0, 101, 1), 'fan_low')
    fan_high = ctrl.Consequent(np.arange(0, 101, 1), 'fan_high')

    # Фаззификация входа
    vehicle_count['few'] = fuzz.trapmf(vehicle_count.universe, [0, 0, 20, 40])
    vehicle_count['medium'] = fuzz.trimf(vehicle_count.universe, [40, 60, 80])
    vehicle_count['many'] = fuzz.trapmf(vehicle_count.universe, [80, 80, 100, 100])

    # Фаззификация выхода
    fan_low['off'] = fuzz.trapmf(fan_low.universe, [0, 0, 20, 40])
    fan_low['on'] = fuzz.trapmf(fan_low.universe, [30, 50, 100, 100])

    fan_high['off'] = fuzz.trapmf(fan_high.universe, [0, 0, 20, 40])
    fan_high['on'] = fuzz.trapmf(fan_high.universe, [30, 60, 100, 100])

    # Правила управления
    rules = [
        ctrl.Rule(vehicle_count['few'], fan_low['off']),
        ctrl.Rule(vehicle_count['medium'], fan_low['on']),
        ctrl.Rule(vehicle_count['many'], fan_high['on']),
        ctrl.Rule(vehicle_count['medium'], fan_high['on']),
    ]

    system = ctrl.ControlSystem(rules)
    return system


# Функция безопасного получения выхода

def safe_get(output, key):
    return output[key] if key in output else 0.0


# Применение управления

def apply_control(vehicle_count, air_quality, analytics, TUNNEL_CTRL):
    sim = ctrl.ControlSystemSimulation(TUNNEL_CTRL)
    sim.input['vehicle_count'] = vehicle_count
    sim.compute()

    fan_low = safe_get(sim.output, 'fan_low')
    fan_high = safe_get(sim.output, 'fan_high')

    # Простая модель изменения качества воздуха
    air_quality += (fan_high + fan_low) * 0.2 - vehicle_count * 0.15
    air_quality = max(0, min(100, air_quality))  # Ограничение

    analytics.log(vehicle_count, air_quality, fan_low, fan_high)

    return air_quality, fan_low, fan_high


# Симуляция

def run_simulation(steps=50):
    tunnel = create_tunnel_ontology()
    analytics = TunnelAnalytics()

    TUNNEL_CTRL = setup_fuzzy_systems(tunnel)

    vehicle_count = random.randint(tunnel['vehicle_range'][0], tunnel['vehicle_range'][1])
    air_quality = random.randint(50, 100)

    print("Запуск симуляции туннеля\n")

    for step in range(1, steps + 1):
        # Случайные изменения потока транспорта
        vehicle_count = min(100, max(0, vehicle_count + random.randint(-10, 15)))

        air_quality, fan_low, fan_high = apply_control(vehicle_count, air_quality, analytics, TUNNEL_CTRL)

        print(
            f"Шаг {step:02d}: "
            f"Транспорт={vehicle_count}, Качество воздуха={air_quality:.1f} | "
            f"Fan Low={fan_low:.0f}, Fan High={fan_high:.0f}"
        )

        time.sleep(0.1)

    analytics.report()
    visualize(analytics.history)


# Визуализация

def visualize(history):
    vehicles = [h['vehicle_count'] for h in history]
    air_quality = [h['air_quality'] for h in history]

    plt.figure(figsize=(10, 4))
    plt.plot(vehicles, label="Количество транспорта")
    plt.plot(air_quality, label="Качество воздуха")
    plt.axhline(80, linestyle='--', color='gray', label="Оптимум")
    plt.legend()
    plt.grid()
    plt.show()

run_simulation()
