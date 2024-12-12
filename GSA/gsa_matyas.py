import random
import numpy
import math
import time
import matplotlib.pyplot as plt
import csv
import os
import tracemalloc

class solution:
    def __init__(self):
        self.best = None
        self.bestIndividual = None
        self.convergence = []
        self.Algorithm = ""
        self.objfname = ""
        self.startTime = ""
        self.endTime = ""
        self.executionTime = 0

# Funkcja obliczająca masy agentów
def massCalculation(fit, PopSize):
    Fmax = max(fit)
    Fmin = min(fit)
    M = numpy.zeros(PopSize)
        
    if Fmax == Fmin:
        M = numpy.ones(PopSize)
    else:
        worst = Fmax
        best = Fmin
        M = (worst - fit) / (worst - best + numpy.finfo(float).eps)
    # Skalowanie mas do zakresu [0.1, 1.0]
    M = (M - M.min()) / (M.max() - M.min() + numpy.finfo(float).eps)
    M = 0.9 * M + 0.1  # Skalowanie do [0.1, 1.0]
    return M

# Funkcja obliczająca stałą grawitacji
def gConstant(l, iters, G0, alpha):
    Gimd = numpy.exp(-alpha * float(l) / iters)
    G = G0 * Gimd
    return G

# Funkcja obliczająca siły i przyspieszenia
def gField(PopSize, dim, pos, M, l, iters, G, ElitistCheck, Rpower, e_constant):
    if ElitistCheck == 1:
        kbest = int(PopSize * (1 - l / iters))
        kbest = max(kbest, 1)  
    else:
        kbest = PopSize
                        
    ds = sorted(range(len(M)), key=lambda k: M[k], reverse=True)
            
    Force = numpy.zeros((PopSize, dim))
            
    for r in range(PopSize):
        for ii in range(kbest):
            z = ds[ii]
            if z != r:
                x = pos[r, :]
                y = pos[z, :]
                R = numpy.linalg.norm(x - y) + e_constant
                for k in range(dim):
                    randnum = random.random()
                    # Obliczanie siły z uwzględnieniem masy agenta r i e_constant
                    Force[r, k] += randnum * M[r] * M[z] * (pos[z, k] - pos[r, k]) / (R ** Rpower + e_constant)
    # Obliczanie przyspieszenia
    acc = Force * G / (M[:, None] + numpy.finfo(float).eps)
    return acc

# Funkcja aktualizująca pozycje i prędkości agentów
def move(PopSize, dim, pos, vel, acc, lb, ub, GraviPert_Min, GraviPert_Max, VelocityPert_Min, VelocityPert_Max):
    for i in range(PopSize):
        for j in range(dim):
            r1 = random.uniform(VelocityPert_Min, VelocityPert_Max)
            r2 = random.uniform(GraviPert_Min, GraviPert_Max)
            vel[i, j] = r1 * vel[i, j] + r2 * acc[i, j]
            pos[i, j] = pos[i, j] + vel[i, j]
            # Ograniczenie pozycji
            if pos[i, j] > ub:
                pos[i, j] = ub
                vel[i, j] = 0
            if pos[i, j] < lb:
                pos[i, j] = lb
                vel[i, j] = 0
    return pos, vel

# Główna funkcja algorytmu GSA
def GSA(objf, lb, ub, dim, PopSize, iters, G0=2.0, alpha=20.0, Rpower=2.0, e_constant=0.01, GraviPert_Min=0.2, GraviPert_Max=0.6, VelocityPert_Min=0.0, VelocityPert_Max=1.0, tol=1e-5):
    # Parametry algorytmu GSA
    ElitistCheck = 1
     
    s = solution()
        
    """ Inicjalizacja """
    
    vel = numpy.zeros((PopSize, dim))
    fit = numpy.zeros(PopSize)
    M = numpy.zeros(PopSize)
    gBest = numpy.zeros(dim)
    gBestScore = float("inf")
    
    pos = numpy.random.uniform(lb, ub, (PopSize, dim))
    
    convergence_curve = []
    avg_fitness_curve = []
    diversity_curve = []
    best_positions_history = []
    
    print("GSA optymalizuje funkcję \"" + objf.__name__ + "\"")    
    
    timerStart = time.time() 
    tracemalloc.start()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    
    for l in range(0, iters):
        # Ocena przystosowania
        for i in range(0, PopSize):
            pos[i, :] = numpy.clip(pos[i, :], lb, ub)
            fitness = objf(pos[i, :])
            fit[i] = fitness
                
            if gBestScore > fitness:
                gBestScore = fitness
                gBest = pos[i, :].copy()
        
        # Obliczanie mas agentów
        M = massCalculation(fit, PopSize)

        # Aktualizacja stałej grawitacji G
        G = gConstant(l, iters, G0, alpha)        
        
        # Obliczanie przyspieszenia
        acc = gField(PopSize, dim, pos, M, l, iters, G, ElitistCheck, Rpower, e_constant)
        
        # Aktualizacja pozycji i prędkości
        pos, vel = move(PopSize, dim, pos, vel, acc, lb, ub, GraviPert_Min, GraviPert_Max, VelocityPert_Min, VelocityPert_Max)
        
        convergence_curve.append(gBestScore)
        avg_fitness_curve.append(numpy.mean(fit))
        best_positions_history.append(gBest.copy())

        # Obliczanie różnorodności (średniej odległości między agentami)
        distances = []
        for i in range(PopSize):
            for j in range(i+1, PopSize):
                distance = numpy.linalg.norm(pos[i] - pos[j])
                distances.append(distance)
        avg_distance = numpy.mean(distances)
        diversity_curve.append(avg_distance)
      
        if (l % 10 == 0):
            print('Iteracja ' + str(l+1) + ': Najlepsza wartość = ' + str(gBestScore))
        
        # Kryterium stopu na podstawie tolerancji
        if abs(gBestScore - GLOBAL_MINIMUM) <= tol:
            print(f"\nZnaleziono globalne minimum z tolerancją {tol} w iteracji {l+1}")
            break
    
    timerEnd = time.time()  
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    s.memoryUsage = peak / 10**6  # Zużycie pamięci w MB
    s.iterations = l + 1  
    s.convergence = convergence_curve
    s.avgFitness = avg_fitness_curve[-1]
    s.diversity = diversity_curve[-1]
    s.Algorithm = "GSA"
    s.objectivefunc = objf.__name__
    s.best = gBestScore
    s.bestIndividual = gBest
    s.G0 = G0
    s.alpha = alpha

    # Zapis wyników do pliku CSV
    save_results_to_csv(s, PopSize)



    return s

# Funkcja zapisująca wyniki do pliku CSV
def save_results_to_csv(s, PopSize):
    run_number = 1
    filename = 'wyniki_gsa_matyas.csv'
    if os.path.isfile(filename):
        with open(filename, mode='r', newline='', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            run_numbers = [int(row['Numer próby']) for row in reader if row['Numer próby'].isdigit()]
            if run_numbers:
                run_number = max(run_numbers) + 1

    results = {
        'Numer próby': run_number,
        'Liczba agentów (N)': PopSize,
        'Początkowa stała grawitacji (G0)': s.G0,
        'Parametr alpha': s.alpha,
        'Najlepsza wartość': s.best,
        'Ilość iteracji': s.iterations,
        'Średnia wartość końcowa': s.avgFitness,
        'Średnia różnorodność końcowa': s.diversity,
        'Czas wykonania (s)': s.executionTime,
        'Zużycie pamięci (MB)': s.memoryUsage
    }

    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as csv_file:
        fieldnames = [
            'Numer próby',
            'Liczba agentów (N)',
            'Początkowa stała grawitacji (G0)',
            'Parametr alpha',
            'Najlepsza wartość',
            'Ilość iteracji',
            'Średnia wartość końcowa',
            'Średnia różnorodność końcowa',
            'Czas wykonania (s)',
            'Zużycie pamięci (MB)'
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)
    print("\nWyniki zostały zapisane do pliku 'wyniki_gsa_matyas.csv'.")

# Funkcja tworząca wykres konwergencji
def plot_convergence(best_fitness_history, avg_fitness_history):
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_history, label='Najlepsza wartość')
    plt.plot(avg_fitness_history, label='Średnia wartość')
    plt.xlabel('Iteracja')
    plt.ylabel('Wartość funkcji celu')
    plt.title('Przystosowanie w funkcji iteracji')
    plt.legend()
    plt.grid(True)
    plt.show()

# Funkcja tworząca wykres różnorodności
def plot_diversity(diversity_history):
    plt.figure(figsize=(10, 6))
    plt.plot(diversity_history)
    plt.xlabel('Iteracja')
    plt.ylabel('Średnia odległość między agentami')
    plt.title('Różnorodność populacji w czasie')
    plt.grid(True)
    plt.show()

# Definicja funkcji Matyasa
def matyas(x):
    x1 = x[0]
    x2 = x[1]
    return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2

# Znane globalne minimum dla funkcji Matyasa
GLOBAL_MINIMUM = 0.0

if __name__ == "__main__":
    # Parametry optymalizacji
    objf = matyas
    lb = -10
    ub = 10
    dim = 2
    PopSize = 100
    iters = 1000
    G0 = 50.0          # Początkowa stała grawitacji
    alpha = 30.0      # Parametr alpha kontrolujący spadek G
    Rpower = 2.0
    e_constant = 0.01
    GraviPert_Min = 0.2
    GraviPert_Max = 0.6
    VelocityPert_Min = 0.0
    VelocityPert_Max = 1.0
    tol = 1e-3        # Tolerancja dla kryterium stopu

    # Uruchomienie algorytmu GSA
    s = GSA(objf, lb, ub, dim, PopSize, iters, G0, alpha, Rpower, e_constant, GraviPert_Min, GraviPert_Max, VelocityPert_Min, VelocityPert_Max, tol)

    # Wyświetlenie wyników
    print("\nNajlepsze znalezione rozwiązanie:", s.bestIndividual)
    print("Najlepsza wartość funkcji celu:", s.best)
    print("Czas wykonania GSA:", s.executionTime, "sekund")
    print("Zużycie pamięci:", s.memoryUsage, "MB")
