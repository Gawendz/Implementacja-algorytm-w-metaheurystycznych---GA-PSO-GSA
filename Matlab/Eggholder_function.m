% Zakres dla zmiennych
x1 = linspace(-512, 512, 100); % Zakres dla zmiennej x1
x2 = linspace(-512, 512, 100); % Zakres dla zmiennej x2

% Tworzenie siatki wartości
[X1, X2] = meshgrid(x1, x2);

% Obliczanie wartości funkcji Eggholder
f = -(X2 + 47) .* sin(sqrt(abs(X2 + X1/2 + 47))) - ...
     X1 .* sin(sqrt(abs(X1 - (X2 + 47))));

% Rysowanie wykresu 3D z siatką
figure;
surf(X1, X2, f);

% Dostosowanie wyglądu wykresu
colormap(jet); % Kolorystyka
colorbar; % Pasek kolorów
title('Eggholder Function', 'FontSize', 10, 'FontWeight', 'bold');
xlabel('x1', 'FontSize', 12);
ylabel('x2', 'FontSize', 12);
zlabel('f(x1, x2)', 'FontSize', 12);

% Widok 3D z siatką
grid on; % Włączenie siatki
view(150, 45); % Ustawienie widoku 3D
shading faceted; % Wyświetlanie krawędzi siatki
