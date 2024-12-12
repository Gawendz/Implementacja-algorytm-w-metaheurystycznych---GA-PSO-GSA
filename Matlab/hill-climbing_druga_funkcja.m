% Definicja głównej funkcji optymalizacyjnej dla hill-climbing szukającego minimum
function hill_climbing_minimum_multimodal()
    % Ustal siatkę wartości dla poziomic
    x = -2:0.1:2;
    y = -2:0.1:2;
    [X, Y] = meshgrid(x, y);
    Z = -exp(-((X-1).^2 + (Y-1).^2)) - 0.5 * exp(-((X+1).^2 + (Y+1).^2));
    
    % Funkcja celu
    f = @(x, y) -exp(-((x - 1)^2 + (y - 1)^2)) - 0.5 * exp(-((x + 1)^2 + (y + 1)^2));
    
    % Punkt początkowy
    x = -0.8; % zmień, aby zobaczyć wpływ na wynik
    y = -0.4;  % zmień, aby zobaczyć wpływ na wynik
    
    % Parametry algorytmu
    step_size = 0.1; % Wielkość kroku
    tolerance = 1e-6; % Tolerancja do zatrzymania
    max_iter = 1000;  % Maksymalna liczba iteracji
    
    % Przechowywanie ścieżki do wizualizacji
    x_history = x;
    y_history = y;
    
    for iter = 1:max_iter
        % Aktualna wartość funkcji celu
        current_value = f(x, y);
        
        % Wygenerowanie sąsiednich punktów (w ośmiu kierunkach)
        neighbors = [
            x + step_size, y;
            x - step_size, y;
            x, y + step_size;
            x, y - step_size;
            x + step_size, y + step_size;
            x - step_size, y + step_size;
            x + step_size, y - step_size;
            x - step_size, y - step_size
        ];
        
        % Sprawdzanie wartości funkcji celu w sąsiednich punktach
        best_value = current_value;
        best_neighbor = [x, y];
        
        for i = 1:size(neighbors, 1)
            neighbor_value = f(neighbors(i, 1), neighbors(i, 2));
            if neighbor_value < best_value % Szukamy minimum
                best_value = neighbor_value;
                best_neighbor = neighbors(i, :);
            end
        end
        
        % Zatrzymanie, jeśli różnica jest mniejsza niż tolerancja
        if abs(best_value - current_value) < tolerance
            break;
        end
        
        % Aktualizacja punktu do najlepszego sąsiada
        x = best_neighbor(1);
        y = best_neighbor(2);
        
        % Zapisanie ścieżki
        x_history = [x_history, x];
        y_history = [y_history, y];
    end
    
    % Wyświetlanie wyników
    fprintf('Lokalne minimum osiągnięte w punkcie (%f, %f) po %d iteracjach.\n', x, y, iter);
    fprintf('Wartość funkcji w lokalnym minimum: %f\n', f(x, y));
    
    % Rysowanie ścieżki optymalizacji na poziomicach
    figure;
    contour(X, Y, Z, 50); % Kontur funkcji z 50 poziomami szczegółowości
    hold on;
    plot(x_history, y_history, 'ro-', 'LineWidth', 1.5); % Trajektoria
    plot(1, 1, 'gx', 'MarkerSize', 10, 'LineWidth', 2); % Minimum globalne
    plot(-1, -1, 'bx', 'MarkerSize', 10, 'LineWidth', 2); % Minimum lokalne
    plot(x_history(1), y_history(1), 'ko', 'MarkerSize', 8, 'LineWidth', 2); % Punkt startowy
    xlabel('x');
    ylabel('y');
    title('Trajektoria metody hill-climbing szukającej minimum dla funkcji multimodalnej');
    legend('Poziomice funkcji', 'Trajektoria', 'Minimum globalne', 'Lokalne minimum', 'Punkt startowy');
    hold off;
end
