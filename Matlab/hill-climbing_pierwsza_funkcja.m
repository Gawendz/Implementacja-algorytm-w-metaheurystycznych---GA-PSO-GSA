% Definicja głównej funkcji optymalizacyjnej dla hill-climbing
function hill_climbing()
    % Funkcja celu
    f = @(x, y) x.^2 + y.^2;
    
    % Punkt początkowy
    x = 2; % możesz zmienić na inne wartości
    y = 3; % możesz zmienić na inne wartości
    
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
        
        % Wygenerowanie sąsiednich punktów (w czterech kierunkach)
        neighbors = [
            x + step_size, y;
            x - step_size, y;
            x, y + step_size;
            x, y - step_size
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
    fprintf('Minimum osiągnięte w punkcie (%f, %f) po %d iteracjach.\n', x, y, iter);
    fprintf('Wartość funkcji w minimum: %f\n', f(x, y));
    
    % Rysowanie ścieżki optymalizacji
    [X, Y] = meshgrid(-3:0.1:3, -3:0.1:3);
    Z = X.^2 + Y.^2;
    contour(X, Y, Z, 50);
    hold on;
    plot(x_history, y_history, 'ro-'); % Trajektoria
    plot(0, 0, 'bx', 'MarkerSize', 10, 'LineWidth', 2); % Minimum globalne
    plot(x_history(1), y_history(1), 'ko', 'MarkerSize', 6, 'LineWidth', 1); % Punkt startowy (mniejsze kółko)
    xlabel('x');
    ylabel('y');
    title('Trajektoria metody hill-climbing dla funkcji z = x^2 + y^2');
    legend('Poziomice funkcji', 'Trajektoria', 'Minimum globalne', 'Punkt startowy');
    hold off;
end
