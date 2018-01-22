function [ ] = fixed_alpha_plots( filename, alpha_idx, fraction, projected )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    load(filename);
    
    %% MSE
    MSE.mean = mean(mse, 4); % Mean over 30 repetitions
    MSE.std  = std(mse, 0, 4);  % Std over 30 repetitions

    figure;
        errorbar(repmat(fraction*100,4,1)', squeeze(MSE.mean(:,alpha_idx,:))', squeeze(MSE.std(:,alpha_idx,:))');
        grid on;
        legend('\mu = 0.8','\mu = 0.85','\mu = 0.9','\mu = 0.95');
        xlabel('% of triplets used')
        ylabel('MSE')
        ylim([-2 7])
        xlim([0 1])
        if projected
            title('Projected');
%             matlab2tikz('MSE_fraction_projected.tex');
        else
            title('Not projected');
%             matlab2tikz('MSE_fraction_notprojected.tex');
        end
        
    %% Triplet violations
    TV.mean = mean(triplet_violations, 4); % Mean over 30 repetitions
    TV.std  = std(triplet_violations, 0, 4);  % Std over 30 repetitions

    figure;
        errorbar(repmat(fraction*100,4,1)', squeeze(TV.mean(:,alpha_idx,:))', squeeze(TV.std(:,alpha_idx,:))');
        grid on;
        legend('\mu = 0.8','\mu = 0.85','\mu = 0.9','\mu = 0.95');
        xlabel('% of triplets used')
        ylabel('Triplet violations')
        if projected
            title('Projected')
        else
            title('Not projected')
        end
end

