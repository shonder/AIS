import numpy as np, random, itertools, pandas as pd, matplotlib.pyplot as plt

random.seed(42); np.random.seed(42)

crops = [
    {"name": "Wheat", "yield_t_per_ha": 4.0, "cost_per_ha": 200},
    {"name": "Corn",  "yield_t_per_ha": 6.0, "cost_per_ha": 300},
    {"name": "Soy",   "yield_t_per_ha": 2.8, "cost_per_ha": 220},
    {"name": "Barley","yield_t_per_ha": 3.5, "cost_per_ha": 180},
]
k = len(crops)
N = 8
field_multipliers = np.random.uniform(0.75, 1.25, size=N)
yields = np.zeros((N, k))
costs = np.array([c["cost_per_ha"] for c in crops])
base_yields = np.array([c["yield_t_per_ha"] for c in crops])
for i in range(N):
    yields[i, :] = base_yields * field_multipliers[i]

w_y = 0.7; w_c = 0.3

def evaluate_assignment(assign):
    total_yield = sum(yields[i, assign[i]] for i in range(N))
    total_cost = sum(costs[assign[i]] for i in range(N))
    min_yield = sum(yields[i, :].min() for i in range(N))
    max_yield = sum(yields[i, :].max() for i in range(N))
    min_cost = N * costs.min()
    max_cost = N * costs.max()
    norm_y = (total_yield - min_yield) / (max_yield - min_yield) if max_yield>min_yield else 0.0
    norm_c = (total_cost - min_cost) / (max_cost - min_cost) if max_cost>min_cost else 0.0
    fitness = w_y * norm_y - w_c * norm_c
    return {"total_yield": total_yield, "total_cost": total_cost, "fitness": fitness}

# Brute force
all_assignments = list(itertools.product(range(k), repeat=N))
best_brute = None; best_brute_val = -1e9
for a in all_assignments:
    res = evaluate_assignment(a)
    if res["fitness"] > best_brute_val:
        best_brute_val = res["fitness"]
        best_brute = {"assign": a, **res}

# GA operators
def one_point_crossover(p1, p2):
    pt = random.randint(1, N-1)
    return np.concatenate([p1[:pt], p2[pt:]])
def two_point_crossover(p1, p2):
    a = random.randint(1, N-2)
    b = random.randint(a+1, N-1)
    c = p1.copy(); c[a:b] = p2[a:b]; return c
def uniform_crossover(p1, p2):
    mask = np.random.randint(0,2,size=N).astype(bool)
    c = p1.copy(); c[mask] = p2[mask]; return c

def mut_random_reset(child, mut_rate=0.05):
    for i in range(N):
        if random.random() < mut_rate:
            child[i] = random.randrange(k)
    return child
def mut_swap(child, mut_rate=0.05):
    if random.random() < mut_rate:
        i,j = random.sample(range(N),2)
        child[i], child[j] = child[j], child[i]
    return child
def mut_creep(child, mut_rate=0.05):
    for i in range(N):
        if random.random() < mut_rate:
            if random.random() < 0.5: child[i] = (child[i] + 1) % k
            else: child[i] = (child[i] - 1) % k
    return child

crossover_methods = {"one_point": one_point_crossover, "two_point": two_point_crossover, "uniform": uniform_crossover}
mutation_methods = {"random_reset": mut_random_reset, "swap": mut_swap, "creep": mut_creep}

def run_ga(pop_size=60, generations=80, cx_name="one_point", mut_name="random_reset", mut_rate=0.06, tournament_k=3):
    pop = [np.array([random.randrange(k) for _ in range(N)]) for _ in range(pop_size)]
    best_history = []
    cx_func = crossover_methods[cx_name]; mut_func = mutation_methods[mut_name]
    for gen in range(generations):
        fitnesses = [evaluate_assignment(ind)["fitness"] for ind in pop]
        best_idx = int(np.argmax(fitnesses))
        best_history.append(fitnesses[best_idx])
        new_pop = []
        while len(new_pop) < pop_size:
            contenders = random.sample(range(pop_size), tournament_k)
            p1 = pop[max([(fitnesses[i], i) for i in contenders])[1]]
            contenders = random.sample(range(pop_size), tournament_k)
            p2 = pop[max([(fitnesses[i], i) for i in contenders])[1]]
            child = cx_func(p1, p2)
            child = mut_func(child.copy(), mut_rate=mut_rate)
            new_pop.append(child)
        pop = new_pop
    fitnesses = [evaluate_assignment(ind)["fitness"] for ind in pop]
    best_idx = int(np.argmax(fitnesses)); best_ind = pop[best_idx]; best_eval = evaluate_assignment(best_ind)
    return {"best_ind": best_ind, "best_eval": best_eval, "history": best_history}

# Experiments
results = []; avg_history_per_combo = {}
runs_per_combo = 3; generations = 80
for cx_name in crossover_methods.keys():
    for mut_name in mutation_methods.keys():
        histories = []; runs_out = []
        for r in range(runs_per_combo):
            out = run_ga(pop_size=60, generations=generations, cx_name=cx_name, mut_name=mut_name, mut_rate=0.06)
            histories.append(out["history"]); runs_out.append(out)
        avg_hist = np.mean(histories, axis=0)
        avg_history_per_combo[(cx_name, mut_name)] = avg_hist
        best_run = max(runs_out, key=lambda x: x["best_eval"]["fitness"])
        results.append({
            "crossover": cx_name,
            "mutation": mut_name,
            "best_fitness": best_run["best_eval"]["fitness"],
            "best_yield": best_run["best_eval"]["total_yield"],
            "best_cost": best_run["best_eval"]["total_cost"],
            "best_assignment": list(best_run["best_ind"])
        })

df_results = pd.DataFrame(results)
brute_row = {
    "crossover": "brute_force", "mutation": "",
    "best_fitness": best_brute["fitness"],
    "best_yield": best_brute["total_yield"],
    "best_cost": best_brute["total_cost"],
    "best_assignment": list(best_brute["assign"])
}
df_results = pd.concat([df_results, pd.DataFrame([brute_row])], ignore_index=True)

# Plots
plt.figure(figsize=(10,6))
for (cx,mu), hist in avg_history_per_combo.items():
    plt.plot(range(1, generations+1), hist, label=f"{cx}/{mu}")
plt.xlabel("Generation"); plt.ylabel("Average best fitness")
plt.title("GA: Average best fitness over generations")
plt.legend(loc="best", fontsize="small"); plt.grid(True); plt.tight_layout(); plt.show()

plt.figure(figsize=(10,6))
sorted_df = df_results.sort_values(by="best_fitness", ascending=False).reset_index(drop=True)
labels = [f"{r['crossover']}/{r['mutation']}" if r['mutation']!="" else "brute_force" for _,r in sorted_df.iterrows()]
plt.bar(labels, sorted_df["best_fitness"])
plt.xticks(rotation=45, ha="right"); plt.ylabel("Best fitness"); plt.title("Best fitness: GA combos vs brute force"); plt.tight_layout(); plt.show()

print("Results table:")
print(df_results)
print("\nBrute-force best solution found:")
print(f"  Fitness = {best_brute['fitness']:.6f}, Total yield = {best_brute['total_yield']:.3f} t, Total cost = ${best_brute['total_cost']:.2f}")
print("Crop names per field:")
for i, crop_idx in enumerate(best_brute["assign"]):
    print(f"  Field {i+1}: {crops[crop_idx]['name']} (yield {yields[i,crop_idx]:.2f} t, cost ${costs[crop_idx]:.2f})")
