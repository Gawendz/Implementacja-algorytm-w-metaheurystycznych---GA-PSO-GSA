% Zakres dla zmiennych
x = linspace(-10, 10, 100); % Zakres dla zmiennej x
y = linspace(-10, 10, 100); % Zakres dla zmiennej y

% Tworzenie siatki wartości
[X, Y] = meshgrid(x, y);

% Obliczanie wartości funkcji
Z = X.^2 + Y.^2;

% Rysowanie wykresu 3D z siatką
figure;
surf(X, Y, Z);

% Dostosowanie wyglądu wykresu
colormap(jet); % Kolorystyka
colorbar; % Pasek kolorów
title('Function: z = x^2 + y^2', 'FontSize', 10, 'FontWeight', 'bold');
xlabel('x', 'FontSize', 12);
ylabel('y', 'FontSize', 12);
zlabel('z', 'FontSize', 12);

% Widok 3D z siatką
grid on; % Włączenie siatki
view(150, 45); % Ustawienie widoku 3D
shading faceted; % Wyświetlanie krawędzi siatki
