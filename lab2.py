import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


def main():
    # Универсум
    x = np.linspace(0, 50, 401)

    print("Введите параметры треугольной функции для множества A (комфортная температура)")
    a1, b1, c1 = map(float, input("a b c: ").split())

    print("\nВведите параметры треугольной функции для множества B (жаркая температура)")
    a2, b2, c2 = map(float, input("a b c: ").split())

    # Нечёткие множества
    comfort = fuzz.trimf(x, [a1, b1, c1])
    hot = fuzz.trimf(x, [a2, b2, c2])

    # Объединение (OR)
    union = np.fmax(comfort, hot)

    # Чёткое значение
    x_inp = float(input("\nВведите температуру (°C): "))

    mu_comfort = fuzz.interp_membership(x, comfort, x_inp)
    mu_hot = fuzz.interp_membership(x, hot, x_inp)
    mu_union = max(mu_comfort, mu_hot)

    print("\n--- Результат ---")
    print(f"Степень принадлежности 'Комфортно': {mu_comfort:.2f}")
    print(f"Степень принадлежности 'Жарко': {mu_hot:.2f}")
    print(f"Объединение (OR): {mu_union:.2f}")

    # График
    plt.plot(x, comfort, label="Комфортная температура")
    plt.plot(x, hot, label="Жаркая температура")
    plt.plot(x, union, label="Объединение (OR)", linestyle="--", linewidth=2)

    plt.scatter([x_inp], [mu_comfort])
    plt.scatter([x_inp], [mu_hot])
    plt.scatter([x_inp], [mu_union], color="black")

    plt.title("Объединение нечётких множеств (OR)")
    plt.xlabel("Температура (°C)")
    plt.ylabel("Степень принадлежности")
    plt.legend()
    plt.grid(True)
    plt.show()

main()
