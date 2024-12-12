% Define the grid for x and y values
x = -2:0.1:2;
y = -2:0.1:2;
[X, Y] = meshgrid(x, y);

% Define the function values Z over the grid
Z = -exp(-((X-1).^2 + (Y-1).^2)) - 0.5 * exp(-((X+1).^2 + (Y+1).^2));

% Plot the 3D surface
figure;
mesh(X, Y, Z);
colormap jet; % Choose a color scheme for visibility
shading interp; % Smooth shading
xlabel('x');
ylabel('y');
zlabel('f(x, y)');

% Add title with the function formula
title('3D Plot of f(x, y) = -exp(-((x-1)^2 + (y-1)^2)) - 0.5 * exp(-((x+1)^2 + (y+1)^2))');

% Set viewing angle for better visibility
view(-30, 30);
