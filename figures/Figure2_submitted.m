clear; close all; clc;

projected = true;
alpha = 11;
mu = 0.8:0.05:0.95;
fraction = [0.0005,...
    0.00083405,...
    0.00139128,...
    0.00232079,...
    0.00387132,...
    0.00645775,...
    0.0107722,...
    0.0179691,...
    0.0299742,...
    0.05];

if projected
    filename = '../results/results_projected';
    load(filename);
    alpha_idx = [1, 11, 19];
    for i = 1:length(alpha_idx)
        fixed_alpha_plots(filename, alpha_idx(i), fraction, projected);
    end
elseif ~projected
    filename = '../results/results_not_projected';
    load(filename);
    alpha_idx = [1, 11, 19];
    for i = 1:length(alpha_idx)
        fixed_alpha_plots(filename, alpha_idx(i), fraction, projected);
    end
end


