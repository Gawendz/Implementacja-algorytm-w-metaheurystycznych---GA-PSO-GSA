import numpy as np
import random
import matplotlib.pyplot as plt
import time
import tracemalloc
import csv
import os

def matyas_function(x):
    x1 = x[0]
    x2 = x[1]
    return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2

# Parametry algorytmu genetycznego
POPULATION_SIZE = 100 
CHROMOSOME_LENGTH = 2  
X_BOUND = [-10, 10]    
CROSSOVER_RATE = 1.0 
MUTATION_RATE = 0.05
ELITISM_COUNT = 10 
GLOBAL_MINIMUM = 0  
TOLERANCE = 1e-3 

def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        individual = np.random.uniform(X_BOUND[0], X_BOUND[1], CHROMOSOME_LENGTH)
        population.append(individual)
    return population

def fitness(individual):
    return matyas_function(individual)  

def selection(population, fitnesses):
    fitnesses = np.array(fitnesses)
    max_fitness = max(fitnesses)
    adjusted_fitnesses = max_fitness - fitnesses + 1e-6  # Unikamy dzielenia przez zero
    total_fitness = sum(adjusted_fitnesses)
    selection_probs = adjusted_fitnesses / total_fitness
    selected_indices = np.random.choice(range(len(population)), size=len(population), p=selection_probs)
    selected = [population[i] for i in selected_indices]
    return selected

def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        # Krzyżowanie BLX-alpha
        alpha = 0.5
        child1 = np.zeros(CHROMOSOME_LENGTH)
        child2 = np.zeros(CHROMOSOME_LENGTH)
        for i in range(CHROMOSOME_LENGTH):
            x_min = min(parent1[i], parent2[i])
            x_max = max(parent1[i], parent2[i])
            range_ = x_max - x_min
            lower_bound = x_min - alpha * range_
            upper_bound = x_max + alpha * range_
            child1[i] = random.uniform(lower_bound, upper_bound)
            child2[i] = random.uniform(lower_bound, upper_bound)
            # Zapewnienie ograniczeń
            child1[i] = np.clip(child1[i], X_BOUND[0], X_BOUND[1])
            child2[i] = np.clip(child2[i], X_BOUND[0], X_BOUND[1])
        return child1, child2
    else:
        return parent1.copy(), parent2.copy()

def mutate(individual, generation, max_generations):
    mutation_rate = MUTATION_RATE
    for i in range(CHROMOSOME_LENGTH):
        if random.random() < mutation_rate:
            individual[i] = random.uniform(X_BOUND[0], X_BOUND[1])

def genetic_algorithm():
    # Rozpoczęcie pomiaru czasu i zużycia pamięci
    start_time = time.time()
    tracemalloc.start()
    
    population = initialize_population()
    best_individual = None
    best_fitness = float('inf')
    best_fitness_history = []
    avg_fitness_history = []
    avg_distance_history = []
    best_individual_history = []
    
    generation = 0
    max_generations = 1000  # Maksymalna liczba generacji
    
    global_min_found = False

    while not global_min_found:
        # Obliczanie przystosowania
        fitnesses = [fitness(individual) for individual in population]
    
        # Aktualizacja najlepszego osobnika
        min_fitness = min(fitnesses)
        min_index = fitnesses.index(min_fitness)
        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_individual = population[min_index].copy()
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(sum(fitnesses) / POPULATION_SIZE)
        best_individual_history.append(best_individual.copy())
    
        # Obliczanie średniej odległości między osobnikami
        distances = []
        for i in range(POPULATION_SIZE):
            for j in range(i+1, POPULATION_SIZE):
                distance = np.linalg.norm(population[i] - population[j])
                distances.append(distance)
        avg_distance = sum(distances) / len(distances)
        avg_distance_history.append(avg_distance)
    
        if (generation+1) % 10 == 0:
            print(f"Generacja {generation+1}: Najlepsze przystosowanie = {best_fitness}, Średnia odległość = {avg_distance}")
    
        # Sprawdzenie, czy znaleziono globalne minimum z zadaną dokładnością
        if abs(best_fitness - GLOBAL_MINIMUM) <= TOLERANCE:
            end_time_global = time.time()
            time_to_global_min = end_time_global - start_time
            current, peak = tracemalloc.get_traced_memory()
            memory_at_global_min = peak / 10**6  # Konwersja na MB
            generations_to_global_min = generation + 1  # Generacje liczone od 1
            print(f"\nZnaleziono globalne minimum w generacji {generations_to_global_min}!")
            print(f"Czas do znalezienia globalnego minimum: {time_to_global_min:.2f} sekund")
            print(f"Zużycie pamięci przy znalezieniu globalnego minimum: {memory_at_global_min:.2f} MB\n")
            global_min_found = True
            break  

        if generation >= max_generations:
            print("\nNie znaleziono globalnego minimum w zadanej liczbie generacji.\n")
            break
    
        # Elityzm
        elite_indices = np.argsort(fitnesses)[:ELITISM_COUNT]
        elites = [population[i] for i in elite_indices]
    
        # Selekcja
        selected = selection(population, fitnesses)
    
        # Krzyżowanie
        next_population = []
        for i in range(0, POPULATION_SIZE - ELITISM_COUNT, 2):
            parent1 = selected[i]
            parent2 = selected[(i+1)%POPULATION_SIZE]
            child1, child2 = crossover(parent1, parent2)
            next_population.extend([child1, child2])
    
        # Mutacja
        for individual in next_population:
            mutate(individual, generation, max_generations)
    
        # Dodanie elity do populacji
        next_population.extend(elites)
        population = next_population[:POPULATION_SIZE]  # Zapewnienie rozmiaru populacji

        generation +=1

    # Zakończenie pomiaru czasu i zużycia pamięci
    end_time = time.time()
    elapsed_time = end_time - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nCzas wykonania: {elapsed_time:.2f} sekund")
    print(f"Zużycie pamięci: {peak / 10**6:.2f} MB")
    
    print(f"\nNajlepszy osobnik: {best_individual}, Przystosowanie: {best_fitness}")
    
    # Wykres przystosowania w funkcji generacji
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_history, label='Najlepsze przystosowanie')
    plt.plot(avg_fitness_history, label='Średnie przystosowanie')
    plt.xlabel('Generacja')
    plt.ylabel('Przystosowanie')
    plt.title('Przystosowanie w funkcji generacji')
    plt.legend()
    plt.show()
    
    # Wykres średniej odległości w funkcji generacji
    plt.figure(figsize=(10, 6))
    plt.plot(avg_distance_history, label='Średnia odległość')
    plt.xlabel('Generacja')
    plt.ylabel('Średnia odległość między osobnikami')
    plt.title('Różnorodność populacji w funkcji generacji')
    plt.legend()
    plt.show()
    
    # Generowanie siatki dla poziomic
    xlist = np.linspace(X_BOUND[0], X_BOUND[1], 400)
    ylist = np.linspace(X_BOUND[0], X_BOUND[1], 400)
    X_mesh, Y_mesh = np.meshgrid(xlist, ylist)
    Z = matyas_function([X_mesh, Y_mesh])
    
    # Wizualizacja trajektorii najlepszego osobnika z poziomicami
    best_individuals = np.array(best_individual_history)
    plt.figure(figsize=(10, 6))
    plt.contour(X_mesh, Y_mesh, Z, levels=50, cmap='viridis')
    plt.plot(best_individuals[:, 0], best_individuals[:, 1], linestyle='-', color='red', alpha=0.7, label='Trajektoria')
    plt.scatter(best_individuals[0, 0], best_individuals[0, 1], color='green', marker='o', s=100, label='Początek')
    plt.scatter(best_individuals[-1, 0], best_individuals[-1, 1], color='blue', marker='X', s=100, label='Koniec')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Trajektoria najlepszego osobnika')
    plt.grid(True)
    plt.legend()
    plt.show()
    
   

    # Przygotowanie danych do zapisu
    run_number = 1
    if os.path.isfile('wyniki_GA_matyas.csv'):
        with open('wyniki_GA_matyas.csv', mode='r', newline='', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            run_numbers = [int(row['Numer próby']) for row in reader if row['Numer próby'].isdigit()]
            if run_numbers:
                run_number = max(run_numbers) + 1

    results = {
        'Numer próby': run_number,
        'Liczba populacji początkowej': POPULATION_SIZE,
        'Prawdopodobieństwo krzyżowania': CROSSOVER_RATE,
        'Prawdopodobieństwo mutacji': MUTATION_RATE,
        'Liczba elityzmu': ELITISM_COUNT,
        'Najlepsze przystosowanie': best_fitness,
        'Ilość iteracji': generation + 1,
        'Średnie przystosowanie': avg_fitness_history[-1],
        'Średnia odległość końcowa': avg_distance_history[-1],
        'Czas wykonania (s)': elapsed_time,
        'Zużycie pamięci (MB)': peak / 10**6
    }

    # Zapis wyników do pliku CSV
    file_exists = os.path.isfile('wyniki_GA_matyas.csv')
    with open('wyniki_GA_matyas.csv', mode='a', newline='', encoding='utf-8') as csv_file:
        fieldnames = [
            'Numer próby',
            'Liczba populacji początkowej',
            'Prawdopodobieństwo krzyżowania',
            'Prawdopodobieństwo mutacji',
            'Liczba elityzmu',
            'Najlepsze przystosowanie',
            'Ilość iteracji',
            'Średnie przystosowanie',
            'Średnia odległość końcowa',
            'Czas wykonania (s)',
            'Zużycie pamięci (MB)'
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)
    print("\nWyniki zostały zapisane do pliku 'wyniki_GA_matyas.csv'.")

if __name__ == "__main__":
    genetic_algorithm()
