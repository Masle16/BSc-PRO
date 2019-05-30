%% Read data
x = textread('x.txt');
y = textread('y.txt');
z = textread('z.txt');

x_mean = textread('x_mean.txt');
y_mean = textread('y_mean.txt');
z_mean = textread('z_mean.txt');

x_mean_keras = textread('x_mean_keras.txt');
y_mean_keras = textread('y_mean_keras.txt');
z_mean_keras = textread('z_mean_keras.txt');

%% Plot for original data versus our mean image subtraction
figure(1)
scatter3(x_mean, y_mean, z_mean)
%scatter3(x,y,z)
hold on
%scatter3(x_mean, y_mean, z_mean)
legend('Images pixels', 'Image pixels subtracted mean image')
title('All downscaled images(16x16) pixel values versus images subtracted with mean image')
xlabel('Red')
ylabel('Green')
zlabel('Blue')
hold off
%% Plot for original data versus keras mean RGB subtraction
figure(2)
scatter3(x_mean_keras, y_mean_keras, z_mean_keras)
%scatter3(x,y,z)
hold on

legend('Images pixels', 'Images pixels subtracted mean rgb ')
title('All downscaled images(16x16) pixel values versus images subtracted with mean rgb values')
xlabel('Red')
ylabel('Green')
zlabel('Blue')
hold off

%% Plot whitening versus not whitening
figure(3)
scatter3(x,y,z)