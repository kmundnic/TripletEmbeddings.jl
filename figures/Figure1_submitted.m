clear; close all; clc;

load ../Results/results_not_projected.mat
MSE.notprojected.mean = mean(mse, 4);
MSE.notprojected.std  = std(mse, 0, 4);

TV.notprojected.mean = mean(triplet_violations, 4);
TV.notprojected.std = std(triplet_violations, 0, 4);

load ../Results/results_projected.mat
MSE.projected.mean = mean(mse, 4);
MSE.projected.std  = std(mse, 0, 4);

TV.projected.mean = mean(triplet_violations, 4);
TV.projected.std = std(triplet_violations, 0, 4);

alpha = 10;
mu = 0.8:0.05:0.95;
fractions = 5*logspace(-4,-2,10);
[MU, FRACTION] = meshgrid(mu, fractions);

figure;
surf(MU, FRACTION, squeeze(TV.projected.mean(:,10,:))');
xlabel('\mu');
ylabel('Fraction of triplets')
ax.YTick = fractions*100;
zlabel('Triplet violations');
view(140,20)
% matlab2tikz('Figure3.tex');
