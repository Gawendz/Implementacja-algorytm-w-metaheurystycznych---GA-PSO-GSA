% Definicja parametrów
m = 10; % Parametr funkcji Michalewicza
x1 = linspace(0, pi, 100); % Zakres dla zmiennej x1
x2 = linspace(0, pi, 100); % Zakres dla zmiennej x2

% Tworzenie siatki wartości
[X1, X2] = meshgrid(x1, x2);

% Obliczanie wartości funkcji Michalewicza
f = -(sin(X1) .* (sin((1 * X1.^2) / pi)).^(2 * m) + ...
      sin(X2) .* (sin((2 * X2.^2) / pi)).^(2 * m));

% Rysowanie wykresu 3D z siatką
figure;
surf(X1, X2, f);

% Dostosowanie wyglądu wykresu
colormap(jet); % Kolorystyka
colorbar; % Pasek kolorów
title('Michalewicz Function', 'FontSize', 10, 'FontWeight', 'bold');
xlabel('x1', 'FontSize', 12);
ylabel('x2', 'FontSize', 12);
zlabel('f(x1, x2)', 'FontSize', 12);

% Widok 3D z siatką
grid on; % Włączenie siatki
view(150, 45); % Ustawienie widoku 3D
shading faceted; % Wyświetlanie krawędzi siatki
