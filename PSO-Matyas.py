import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time
import tracemalloc
import csv
import os

def matyas_function(x):
    x1 = x[0]
    x2 = x[1]
    return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2


# Parametry PSO
NUM_PARTICLES = 100       # Liczba cząstek w roju
NUM_DIMENSIONS = 2       # Liczba wymiarów problemu
MAX_ITERATIONS = 1000    # Maksymalna liczba iteracji
INERTIA_WEIGHT = 1    # Waga inercji
COGNITIVE_CONST = 2      # Stała poznawcza (c1)
SOCIAL_CONST = 2         # Stała społeczna (c2)

# Zakresy zmiennych
X_MIN = -10
X_MAX = 10

# Globalne minimum
GLOBAL_MINIMUM = 0       # Globalne minimum funkcji Matyasa
TOLERANCE = 1e-3        # Dokładność rozwiązania


class Particle:
    def __init__(self):
        self.position = np.random.uniform(X_MIN, X_MAX, NUM_DIMENSIONS)
        self.velocity = np.zeros(NUM_DIMENSIONS)
        self.best_position = self.position.copy()
        self.best_value = matyas_function(self.position)
        
    def update_velocity(self, global_best_position):
        r1 = np.random.uniform(0, 1, NUM_DIMENSIONS)
        r2 = np.random.uniform(0, 1, NUM_DIMENSIONS)
        cognitive_component = COGNITIVE_CONST * r1 * (self.best_position - self.position)
        social_component = SOCIAL_CONST * r2 * (global_best_position - self.position)
        self.velocity = INERTIA_WEIGHT * self.velocity + cognitive_component + social_component
        
        # Ograniczenie prędkości
        V_MAX = (X_MAX - X_MIN) * 0.1
        self.velocity = np.clip(self.velocity, -V_MAX, V_MAX)
        
    def update_position(self):
        self.position += self.velocity
        # Zastosowanie ograniczeń
        self.position = np.clip(self.position, X_MIN, X_MAX)
        value = matyas_function(self.position)
        if value < self.best_value:
            self.best_value = value
            self.best_position = self.position.copy()


def pso():
    # Rozpoczęcie pomiaru czasu i zużycia pamięci
    start_time = time.time()
    tracemalloc.start()
    
    # Inicjalizacja cząstek
    swarm = [Particle() for _ in range(NUM_PARTICLES)]
    # Inicjalizacja najlepszego globalnego rozwiązania
    global_best_particle = min(swarm, key=lambda p: p.best_value)
    global_best_position = global_best_particle.best_position.copy()
    global_best_value = global_best_particle.best_value
    
    best_values_history = []
    avg_values_history = []
    diversity_history = []
    iterations = 0
    global_min_found = False
    best_positions_history = []
    
    while iterations < MAX_ITERATIONS and not global_min_found:
        for particle in swarm:
            particle.update_velocity(global_best_position)
            particle.update_position()
        
        # Aktualizacja najlepszego globalnego rozwiązania
        current_best_particle = min(swarm, key=lambda p: p.best_value)
        if current_best_particle.best_value < global_best_value:
            global_best_value = current_best_particle.best_value
            global_best_position = current_best_particle.best_position.copy()
        
        # Zapis historii
        best_values_history.append(global_best_value)
        avg_value = np.mean([p.best_value for p in swarm])
        avg_values_history.append(avg_value)
        best_positions_history.append(global_best_position.copy())
        
        # Obliczanie różnorodności (średniej odległości między cząstkami)
        positions = np.array([p.position for p in swarm])
        distances = []
        for i in range(NUM_PARTICLES):
            for j in range(i+1, NUM_PARTICLES):
                distance = np.linalg.norm(positions[i] - positions[j])
                distances.append(distance)
        avg_distance = np.mean(distances)
        diversity_history.append(avg_distance)
        
        if (iterations + 1) % 10 == 0:
            print(f"Iteracja {iterations + 1}: Najlepsza wartość = {global_best_value}, Średnia odległość = {avg_distance}")
        
        # Kryterium stopu
        if abs(global_best_value - GLOBAL_MINIMUM) <= TOLERANCE:
            end_time_global = time.time()
            time_to_global_min = end_time_global - start_time
            current, peak = tracemalloc.get_traced_memory()
            memory_at_global_min = peak / 10**6  # Konwersja na MB
            iterations_to_global_min = iterations + 1
            print(f"\nZnaleziono globalne minimum w iteracji {iterations_to_global_min}!")
            print(f"Czas do znalezienia globalnego minimum: {time_to_global_min:.6f} sekund")
            print(f"Zużycie pamięci przy znalezieniu globalnego minimum: {memory_at_global_min:.2f} MB\n")
            global_min_found = True
            break
        
        iterations += 1
    
    # Zakończenie pomiaru czasu i zużycia pamięci
    end_time = time.time()
    elapsed_time = end_time - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"\nNajlepsze znalezione rozwiązanie: {global_best_position}")
    print(f"Wartość funkcji celu: {global_best_value}")
    print(f"Czas wykonania: {elapsed_time:.6f} sekund")
    print(f"Zużycie pamięci: {peak / 10**6:.2f} MB")
    
    # Wykres konwergencji
    plt.figure(figsize=(10, 6))
    plt.plot(best_values_history, label='Najlepsza wartość')
    plt.plot(avg_values_history, label='Średnia wartość')
    plt.xlabel('Iteracja')
    plt.ylabel('Wartość funkcji celu')
    plt.title('Przystosowanie w funkcji iteracji')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Wykres różnorodności
    plt.figure(figsize=(10, 6))
    plt.plot(diversity_history)
    plt.xlabel('Iteracja')
    plt.ylabel('Średnia odległość między cząstkami')
    plt.title('Różnorodność roju w czasie')
    plt.grid(True)
    plt.show()
    
    # Analiza konwergencji
    if len(best_values_history) > 1:
        fitness_diffs = [abs(best_values_history[i+1] - best_values_history[i]) for i in range(len(best_values_history)-1)]
        plt.figure(figsize=(10, 6))
        plt.plot(fitness_diffs)
        plt.xlabel('Iteracja')
        plt.ylabel('Zmiana najlepszej wartości')
        plt.title('Analiza konwergencji')
        plt.grid(True)
        plt.show()
    
    # Generowanie siatki dla poziomic
    xlist = np.linspace(X_MIN, X_MAX, 500)
    ylist = np.linspace(X_MIN, X_MAX, 500)
    X_mesh, Y_mesh = np.meshgrid(xlist, ylist)
    Z = matyas_function([X_mesh, Y_mesh])
    
    # Wizualizacja trajektorii najlepszego rozwiązania na tle poziomic
    best_positions = np.array(best_positions_history)
    plt.figure(figsize=(10, 6))
    # Rysowanie poziomic
    plt.contour(X_mesh, Y_mesh, Z, levels=50, cmap='viridis')
    # Rysowanie trajektorii
    plt.plot(best_positions[:, 0], best_positions[:, 1], linestyle='-', color='red', alpha=0.7, label='Trajektoria')
    # Zaznaczenie punktu początkowego
    plt.scatter(best_positions[0, 0], best_positions[0, 1], color='green', marker='o', s=100, label='Początek')
    # Zaznaczenie punktu końcowego
    plt.scatter(best_positions[-1, 0], best_positions[-1, 1], color='blue', marker='X', s=100, label='Koniec')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Trajektoria najlepszego rozwiązania na tle poziomic funkcji Matyasa')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Przygotowanie danych do zapisu
    # Sprawdzanie numeru uruchomienia
    run_number = 1
    if os.path.isfile('wyniki_pso_matyas.csv'):
        with open('wyniki_pso_matyas.csv', mode='r', newline='', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            run_numbers = [int(row['Numer uruchomienia']) for row in reader if row['Numer uruchomienia'].isdigit()]
            if run_numbers:
                run_number = max(run_numbers) + 1
    
    results = {
        'Numer uruchomienia': run_number,
        'Liczba cząstek': NUM_PARTICLES,
        'Waga inercji': INERTIA_WEIGHT,
        'Stała poznawcza (c1)': COGNITIVE_CONST,
        'Stała społeczna (c2)': SOCIAL_CONST,
        'Najlepsza wartość': global_best_value,
        'Po ilu iteracjach znaleziono rozwiązanie': iterations + 1,
        'Średnia wartość końcowa': avg_values_history[-1],
        'Średnia różnorodność końcowa': diversity_history[-1],
        'Czas wykonania (s)': elapsed_time,
        'Zużycie pamięci (MB)': peak / 10**6
    }
    
    # Zapis wyników do pliku CSV
    file_exists = os.path.isfile('wyniki_pso_matyas.csv')
    with open('wyniki_pso_matyas.csv', mode='a', newline='', encoding='utf-8') as csv_file:
        fieldnames = [
            'Numer uruchomienia',
            'Liczba cząstek',
            'Waga inercji',
            'Stała poznawcza (c1)',
            'Stała społeczna (c2)',
            'Najlepsza wartość',
            'Po ilu iteracjach znaleziono rozwiązanie',
            'Średnia wartość końcowa',
            'Średnia różnorodność końcowa',
            'Czas wykonania (s)',
            'Zużycie pamięci (MB)'
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)
    print("\nWyniki zostały zapisane do pliku 'wyniki_pso_matyas.csv'.")


if __name__ == "__main__":
    pso()
