% Zakres dla zmiennych
x1 = linspace(-10, 10, 100); % Zakres dla zmiennej x1
x2 = linspace(-10, 10, 100); % Zakres dla zmiennej x2

% Tworzenie siatki wartości
[X1, X2] = meshgrid(x1, x2);

% Obliczanie wartości funkcji Matyas
f = 0.26 * (X1.^2 + X2.^2) - 0.48 * X1 .* X2;

% Rysowanie wykresu 3D z siatką
figure;
surf(X1, X2, f);

% Dostosowanie wyglądu wykresu
colormap(jet); % Kolorystyka
colorbar; % Pasek kolorów
title('Matyas Function', 'FontSize', 10, 'FontWeight', 'bold');
xlabel('x1', 'FontSize', 12);
ylabel('x2', 'FontSize', 12);
zlabel('f(x1, x2)', 'FontSize', 12);

% Widok 3D z siatką
grid on; % Włączenie siatki
view(150, 45); % Ustawienie widoku 3D
shading faceted; % Wyświetlanie krawędzi siatki
