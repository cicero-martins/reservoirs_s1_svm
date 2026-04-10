%% Codice per ottenere le statistiche tra due csv data files

clear;
clc;

% -------------------------------------------------------------------------
% Loading and parsing data
% ------------------------------------------------------------------------
% load two CSVs: Planet and ADIB (1st-of-month)
% 

% %Rosamarina
% TSVM  = readtable('rosamarina_svm.csv');   % columns: Date, Volume
% TADIB = readtable('rosamarina_adib.csv'); % columns: Date, Volume
% TSVM  = readtable('rosamarina2425_svm.csv');   % columns: Date, Volume
% TADIB = readtable('rosamarina2425_planet.csv'); % columns: Date, Volume
% %TADIB = readtable('rosamarina2425_adib.csv'); % columns: Date, Volume
TSVM  = readtable('rosamarinaSVM.csv');   % columns: Data, area
TADIB = readtable('rosamarinaPlanet.csv'); % columns: Data, area


% % %Ancipa
% % TSVM  = readtable('ancipa_svm.csv');   % columns: Date, Volume
% % TADIB = readtable('ancipa_adib.csv'); % columns: Date, Volume
% TSVM  = readtable('ancipa2425_svm.csv');   % columns: Date, Volume
% TADIB = readtable('ancipa2425_planet.csv'); % columns: Date, Volume
% TSVM  = readtable('ancipaPlanet.csv');   % columns: Data, area
% TADIB = readtable('ancipaSVM.csv'); % columns: Data, area


% %Poma
% TSVM  = readtable('poma_svm.csv');   % columns: Date, Volume
% TADIB = readtable('poma_adib.csv'); % columns: Date, Volume
% TSVM  = readtable('poma2425_svm.csv');   % columns: Date, Volume
% TADIB = readtable('poma2425_planet.csv'); % columns: Date, Volume
% TSVM  = readtable('pomaSVM.csv');   % columns: Data, area
% TADIB = readtable('pomaPlanet.csv'); % columns: Data, area

%Pozzillo
% TSVM  = readtable('pozzillo_svm.csv');   % columns: Date, Volume
% TADIB = readtable('pozzillo_adib.csv'); % columns: Date, Volume
% TSVM  = readtable('pozzillo2425_svm.csv');   % columns: Date, Volume
% TADIB = readtable('pozzillo2425_planet.csv'); % columns: Date, Volume
% TSVM  = readtable('pozzilloSVM.csv');   % columns: Data, area
% TADIB = readtable('pozzilloPlanet.csv'); % columns: Data, area

% % %San Giovanni
% % TSVM  = readtable('sangiovanni_svm.csv');   % columns: Date, Volume
% % TADIB = readtable('sangiovanni_adib.csv'); % columns: Date, Volume
% TSVM  = readtable('sangiovanni2425_planet.csv');   % columns: Date, Volume
% TADIB = readtable('sangiovanni2425_adib.csv'); % columns: Date, Volume


% parse dates (adjust InputFormat if your CSV uses other format)
TSVM.data  = datetime(TSVM.data,  'InputFormat','yyyy-MM-dd');
TADIB.data = datetime(TADIB.data, 'InputFormat','yyyy-MM-dd');


% convert to timetables
TT_svm  = table2timetable(TSVM,  'RowTimes','data');
TT_adib = table2timetable(TADIB, 'RowTimes','data');

% optionally sort and remove duplicate times (average duplicates)
TT_svm = sortrows(TT_svm);
[unqDates,~,ic] = unique(TT_svm.data);
if numel(unqDates) < height(TT_svm)
    avgVol = accumarray(ic, TT_svm.area, [], @mean);
    TT_svm = timetable(unqDates, avgVol, 'VariableNames',{'area'});
end

% interpolating

% Use retime to interpolate SVM and Planet at the ADIB times (linear interpolation)
targetDates = TT_adib.data;                 % 1st of month datetimes
TT_svm_on_adib = retime(TT_svm, targetDates, 'linear');  % numeric interpolation

% now join into one timetable
TT_join = synchronize(TT_adib, TT_svm_on_adib); % columns: Volume_TADIB, Volume_TTSVM

% compute error metrics (omit NaNs)
obs = TT_join.area_TT_adib;
sim = TT_join.area_TT_svm_on_adib;
err = sim - obs;

%% Calculating statistics

% Metrics complemented

function metrics = eval_reservoir_model(obs, sim, time)
% obs   = vetor de observações (ex: AdiB)
% sim   = vetor de simulações (ex: SVM/Planet)
% time  = vetor datetime correspondente
%
% Saída:
% metrics = struct com todas as métricas calculadas

    % --- Remover pares com NaN ---
    valid = ~isnan(obs) & ~isnan(sim);
    obs = obs(valid);
    sim = sim(valid);
    time = time(valid);

    % --- Erros básicos ---
    err  = sim - obs;
    rmse = sqrt(mean(err.^2));
    mae  = mean(abs(err));
    bias = mean(err);
    pbias = 100 * sum(err) / sum(obs);

    % --- NSE ---
    nse = 1 - sum((obs - sim).^2) / sum((obs - mean(obs)).^2);

    % --- KGE (Gupta et al., 2009) ---
    r     = corr(sim, obs);
    alpha = std(sim) / std(obs);
    beta  = mean(sim) / mean(obs);
    kge   = 1 - sqrt((r-1).^2 + (alpha-1).^2 + (beta-1).^2);

    % --- R² ---
    R = corrcoef(obs, sim);
    r2 = R(1,2)^2;

    % --- Índice de concordância (Willmott, 1981) ---
    d = 1 - sum((sim - obs).^2) / sum((abs(sim - mean(obs)) + abs(obs - mean(obs))).^2);

    % --- Erro de pico ---
    [obs_max, idx_obs] = max(obs);
    [sim_max, idx_sim] = max(sim);
    peak_error = 100 * (sim_max - obs_max) / obs_max;

    % --- Erro de tempo (lag em dias) ---
    lag_days = days(time(idx_sim) - time(idx_obs));

    % --- Empacotar ---
    metrics = struct( ...
        'RMSE', rmse, ...
        'MAE', mae, ...
        'Bias', bias, ...
        'PBIAS', pbias, ...
        'NSE', nse, ...
        'KGE', kge, ...
        'R2', r2, ...
        'Index_d', d, ...
        'PeakError_percent', peak_error, ...
        'Lag_days', lag_days ...
    );
end

metrics = eval_reservoir_model(obs, sim, TT_join.data);
disp(metrics)

% Grafico

figure;
plot(TT_join.data, obs, '-o', 'LineWidth', 1.5, 'MarkerSize', 5);
hold on;
plot(TT_join.data, sim, '-s', 'LineWidth', 1.5, 'MarkerSize', 5);
hold off;

xlabel('Data');
ylabel('Volume (hm^3)'); % ajuste a unidade se necessário
title('Série temporal: Observado vs Simulado');
legend({'Observado (AdiB)','Simulado (SVM/Planet)'}, 'Location','best');
grid on;


%% Kling-Gupta Efficiency

function kge_struct = compute_kge(obs, sim)
% COMPUTE_KGE Compute KGE and its components (r, alpha, beta).
%   kge_struct = compute_kge(obs, sim)
%   Inputs:
%     obs - vector of observed values
%     sim - vector of simulated values (same length)
%   Output (struct):
%     r      - Pearson correlation
%     alpha  - sigma_sim / sigma_obs (variability ratio)
%     beta   - mean_sim / mean_obs (bias ratio)
%     KGE    - Kling-Gupta Efficiency
%     n      - number of valid pairs used
%
% Example:
%   k = compute_kge(obs, sim);
%   fprintf('KGE = %.3f (r=%.3f, alpha=%.3f, beta=%.3f)\n', k.KGE, k.r, k.alpha, k.beta);

    % Ensure column vectors
    obs = obs(:);
    sim = sim(:);

    % Remove NaN pairs
    valid = ~isnan(obs) & ~isnan(sim);
    obs = obs(valid);
    sim = sim(valid);
    n = numel(obs);

    if n < 2
        error('Not enough valid data points to compute KGE.');
    end

    % Components
    r = corr(obs, sim);                % Pearson correlation
    sigma_obs = std(obs, 1);           % population std (use 1 for consistency)
    sigma_sim = std(sim, 1);
    mean_obs = mean(obs);
    mean_sim = mean(sim);

    alpha = sigma_sim / sigma_obs;
    beta  = mean_sim / mean_obs;

    % KGE (original formulation)
    KGE = 1 - sqrt( (r-1)^2 + (alpha-1)^2 + (beta-1)^2 );

    % Pack results
    kge_struct = struct( ...
        'r', r, ...
        'alpha', alpha, ...
        'beta', beta, ...
        'KGE', KGE, ...
        'n', n ...
    );

    % Print friendly summary
    fprintf('KGE summary (N=%d):\n', n);
    fprintf('  r (corr)   = %.4f\n', r);
    fprintf('  alpha      = %.4f (sigma_sim/sigma_obs)\n', alpha);
    fprintf('  beta       = %.4f (mean_sim/mean_obs)\n', beta);
    fprintf('  KGE        = %.4f\n', KGE);

    % short interpretation hint
    if KGE >= 0.75
        disp('  Interpretation: very good model performance (KGE >= 0.75).');
    elseif KGE >= 0.5
        disp('  Interpretation: acceptable performance (0.5 <= KGE < 0.75).');
    elseif KGE >= 0
        disp('  Interpretation: poor performance (0 <= KGE < 0.5).');
    else
        disp('  Interpretation: performance worse than using mean (KGE < 0).');
    end
end


k = compute_kge(obs, sim);