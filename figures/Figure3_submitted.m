clear; close all; clc;

alpha = 2:20;

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


fraction = 1; % This is the index for percentage of triplets to be 0.05%

for i = 1:4
    figure(i);
    hold on; grid on;
    errorbar(alpha', squeeze(MSE.notprojected.mean(i,:,fraction))', ...
                            squeeze(MSE.notprojected.std(i,:,fraction))');
    errorbar(alpha', squeeze(MSE.projected.mean(i,:,fraction))', ...
                            squeeze(MSE.projected.std(i,:,fraction))');
    xlabel('\alpha');
    if i == 1
        ylabel('MSE');
    end
    legend('t-STE','\Pi t-STE')
%     matlab2tikz(['Figure3.', num2str(i), '.tex']);
end












