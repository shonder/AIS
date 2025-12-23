import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import product

np.random.seed(42)
random.seed(42)

N_FIELDS = 6
K_CROPS = 5
POP_SIZE = 60
GENERATIONS = 300
MUTATION_RATE = 0.4

crop_names = ["Wheat", "Barley", "Corn", "Sunflower", "Soy"]

# Закупочная цена культур (руб/т)
crop_price = np.array([12000, 10000, 9000, 20000, 18000])

# Урожайность культур на каждом поле (т/га)
yield_matrix = np.random.uniform(
    low=[3, 2.5, 4, 2, 2],
    high=[5, 4, 7, 3, 3.5],
    size=(N_FIELDS, K_CROPS)
)

# ГА
def create_individual():
    return np.random.randint(0, K_CROPS, size=N_FIELDS)

def create_population():
    return [create_individual() for _ in range(POP_SIZE)]

def fitness(individual):
    total_yield = sum(yield_matrix[i, individual[i]] for i in range(N_FIELDS))
    total_cost = sum(
        yield_matrix[i, individual[i]] * crop_price[individual[i]]
        for i in range(N_FIELDS)
    )
    return total_yield / total_cost

# Скрещивания
def one_point_crossover(p1, p2):
    point = random.randint(1, N_FIELDS - 1)
    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])
    return c1, c2

def two_point_crossover(p1, p2):
    a, b = sorted(random.sample(range(N_FIELDS), 2))
    c1, c2 = p1.copy(), p2.copy()
    c1[a:b], c2[a:b] = p2[a:b], p1[a:b]
    return c1, c2

def uniform_crossover(p1, p2):
    mask = np.random.randint(0, 2, N_FIELDS)
    c1 = np.where(mask, p1, p2)
    c2 = np.where(mask, p2, p1)
    return c1, c2

# Мутации
def mutate_random(ind):
    i = random.randint(0, N_FIELDS - 1)
    ind[i] = random.randint(0, K_CROPS - 1)
    return ind

def mutate_shift(ind):
    return np.roll(ind, 1)

def mutate_smart(ind):
    i = random.randint(0, N_FIELDS - 1)
    best_crop = np.argmax(yield_matrix[i] / crop_price)
    ind[i] = best_crop
    return ind

def genetic_algorithm():
    population = create_population()
    best_history = []

    for gen in range(GENERATIONS):
        population = sorted(population, key=fitness, reverse=True)
        best_history.append(fitness(population[0]))

        selected = population[:POP_SIZE // 2]
        offspring = []

        while len(offspring) < POP_SIZE:
            p1, p2 = random.sample(selected, 2)
            choice = random.randint(0, 2)

            if choice == 0:
                c1, c2 = one_point_crossover(p1, p2)
            elif choice == 1:
                c1, c2 = two_point_crossover(p1, p2)
            else:
                c1, c2 = uniform_crossover(p1, p2)

            offspring.extend([c1, c2])

        for i in range(len(offspring)):
            if random.random() < MUTATION_RATE:
                mut = random.randint(0, 2)
                if mut == 0:
                    offspring[i] = mutate_random(offspring[i])
                elif mut == 1:
                    offspring[i] = mutate_shift(offspring[i])
                else:
                    offspring[i] = mutate_smart(offspring[i])

        population = offspring

    population = sorted(population, key=fitness, reverse=True)
    return population[0], best_history

# Полный перебор
def brute_force():
    best_solution = None
    best_fit = -1

    for comb in product(range(K_CROPS), repeat=N_FIELDS):
        ind = np.array(comb)
        f = fitness(ind)
        if f > best_fit:
            best_fit = f
            best_solution = ind

    return best_solution, best_fit

# Запуск
best_ga_solution, history = genetic_algorithm()
best_brute_solution, brute_fit = brute_force()
print("Матрица урожайности (т/га):")
print(np.round(yield_matrix, 2))
print("\nЦены культур (руб/т):")
for i in range(K_CROPS):
    print(f"{crop_names[i]}: {crop_price[i]}")

print("\nЛучшее решение ГА (культура на поле):")
for i in range(N_FIELDS):
    print(f"Поле {i + 1}: {crop_names[best_ga_solution[i]]}")

print("\nFitness (ГА):", fitness(best_ga_solution))

print("\nЛучшее решение полным перебором:")
for i in range(N_FIELDS):
    print(f"Поле {i + 1}: {crop_names[best_brute_solution[i]]}")

print("\nFitness (полный перебор):", brute_fit)

plt.plot(history)
plt.xlabel("Поколение")
plt.ylabel("Лучшее значение fitness")
plt.title("Сходимость генетического алгоритма")
plt.grid(True)
plt.show()
