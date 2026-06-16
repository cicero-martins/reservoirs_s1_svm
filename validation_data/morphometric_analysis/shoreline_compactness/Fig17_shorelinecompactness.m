%% ====================== CONFIG ======================
clear; clc;

% Arquivos de ÁREA (cols: date, areaLago, value) — usaremos 'value'
areaFiles = struct( ...
  'Ancipa',     'area_ancipa_2014-25.csv', ...
  'Poma',       'area_poma_2014-25.csv', ...
  'Pozzillo',   'area_pozzillo_2014-25.csv', ...
  'Rosamarina', 'area_rosamarina_2014-25.csv');

% Arquivos de MÉTRICA (cols: date, value)
metricFiles = struct( ...
  'D',  {{'fractalDimension_ancipa2014-25.csv', ...
          'fractalDimension_poma2014-25.csv', ...
          'fractalDimension_pozzillo2014-25.csv', ...
          'fractalDimension_rosamarina2014-25.csv'}}, ...
  'AP', {{'fractalDimensionAP_ancipa2014-25.csv', ...
          'fractalDimensionAP_poma2014-25.csv', ...
          'fractalDimensionAP_pozzillo2014-25.csv', ...
          'fractalDimensionAP_rosamarina2014-25.csv'}}, ...
  'DL', {{'fractalDimensionDL_ancipa2014-25.csv', ...
          'fractalDimensionDL_poma2014-25.csv', ...
          'fractalDimensionDL_pozzillo2014-25.csv', ...
          'fractalDimensionDL_rosamarina2014-25.csv'}});

reservoirNames = {'Ancipa','Poma','Pozzillo','Rosamarina'};

% Aparência
colors  = lines(numel(reservoirNames));
markers = {'o','s','^','d'};

%% ====================== PLOTS (sem normalização) ======================
% plotUnifiedMetricRaw('D',  'Fractal Dimension (D)', 'D (-)', ...
%     metricFiles.D, areaFiles, reservoirNames, colors, markers);

hSeries = plotUnifiedMetricRaw('AP', 'A/P', 'A/P (m)', ...
    metricFiles.AP, areaFiles, reservoirNames, colors, markers);
print(hSeries, 'AP_vs_area_series', '-dpng', '-r150');

% plotUnifiedMetricRaw('DL', 'DL', 'DL (-)', ...
%     metricFiles.DL, areaFiles, reservoirNames, colors, markers);

%% ====================== FIGURA: A/P MEDIANA vs KGE ======================
% KGE (SVM VV+VH vs PlanetScope) de table3_ablation_metrics.csv
% Ordem: Ancipa, Poma, Pozzillo, Rosamarina
kge_vals  = [0.808, 0.942, 0.969, 0.889];
rmse_vals = [16.7,  13.7,   6.2,   4.5 ];

% A/P mediana (calculada das séries 2014-2025)
ap_medians = [73.6, 173.1, 225.3, 154.3];  % Ancipa, Poma, Pozzillo, Rosamarina

hKGE = plotAPvsKGE(ap_medians, kge_vals, rmse_vals, reservoirNames, colors, markers);
print(hKGE, 'AP_vs_KGE', '-dpng', '-r150');

%% ====================== FUNÇÕES LOCAIS ======================
function hFig = plotUnifiedMetricRaw(metricKey, metricTitle, yLabel, fileList, areaFiles, rnames, colors, markers)
  hFig = figure('Name', metricKey, 'Visible','off'); hold on; grid on; box on;

  for i = 1:numel(rnames)
    r = rnames{i};
    mfile = fileList{i};
    afile = areaFiles.(r);

    [xPct, yMetric] = loadPairsAreaMetric(afile, mfile);  % x em %, y bruto (sem normalização)

    % Scatter (sem contorno, menor, cor única por reservatório)
    s = scatter(xPct, yMetric, 16, 'filled', ...
            'Marker', markers{mod(i-1,numel(markers))+1}, ...
            'MarkerFaceColor', colors(i,:), ...
            'MarkerEdgeColor', 'none', ...
            'MarkerFaceAlpha', 0.4, ...
            'DisplayName', r);
    % Transparência (se suportado)
    try, set(s,'MarkerFaceAlpha',0.8); catch, end

    % Trendline por reservatório (linear), mesma cor do marcador
    mask = isfinite(xPct) & isfinite(yMetric);
    if nnz(mask) >= 2
      p = polyfit(xPct(mask), yMetric(mask), 1);
      xx = linspace(0,100,200);
      yy = polyval(p, xx);
      plot(xx, yy, '-', 'Color', colors(i,:), 'LineWidth', 1.5, 'HandleVisibility','off');
      % (Opcional) imprimir coeficientes no console
      fprintf('%s | %s: slope = %.4g, intercept = %.4g\n', metricKey, r, p(1), p(2));
    end
  end

  xlabel('% of max area');
  ylabel(yLabel);
  title(sprintf('All reservoirs — %s vs %% of max area (raw ranges)', metricTitle));
  legend('Location','bestoutside');
  xlim([0 100]);  % X é percentual
end

function hFig = plotAPvsKGE(ap, kge, rmse, rnames, colors, markers)
  % Scatter plot: median A/P (x) vs KGE (y) and RMSE (secondary axis)
  % Re-order by A/P for clean plot
  [ap_sorted, idx] = sort(ap);
  kge_sorted  = kge(idx);
  rmse_sorted = rmse(idx);
  names_sorted = rnames(idx);

  hFig = figure('Name','AP_vs_KGE','Position',[100 100 700 420],'Visible','off'); hold on; grid on; box on;

  for i = 1:numel(ap_sorted)
    scatter(ap_sorted(i), kge_sorted(i), 120, ...
      'Marker', markers{mod(i-1,numel(markers))+1}, ...
      'MarkerFaceColor', colors(idx(i),:), ...
      'MarkerEdgeColor', 'k', 'LineWidth', 0.8, ...
      'DisplayName', names_sorted{i});
    text(ap_sorted(i)+3, kge_sorted(i)-0.005, ...
      sprintf('%s\n(RMSE=%.1f ha)', names_sorted{i}, rmse_sorted(i)), ...
      'FontSize', 8, 'Color', colors(idx(i),:));
  end

  % Linear regression KGE ~ A/P
  p = polyfit(ap_sorted, kge_sorted, 1);
  xx = linspace(min(ap_sorted)-10, max(ap_sorted)+10, 200);
  yy = polyval(p, xx);
  plot(xx, yy, 'k--', 'LineWidth', 1.2, 'HandleVisibility','off');

  % R^2
  kge_hat = polyval(p, ap_sorted);
  ss_res = sum((kge_sorted - kge_hat).^2);
  ss_tot = sum((kge_sorted - mean(kge_sorted)).^2);
  r2 = 1 - ss_res/ss_tot;
  text(0.05, 0.95, sprintf('Linear fit: KGE = %.4g·A/P + %.4g\nR² = %.3f', ...
    p(1), p(2), r2), 'Units','normalized', ...
    'VerticalAlignment','top', 'FontSize', 9, ...
    'BackgroundColor','white', 'EdgeColor','k');

  xlim([0 270]);  ylim([0.75 1.0]);
  xlabel('Median A/P ratio (m)');
  ylabel('KGE — SVM VV+VH vs. PlanetScope');
  title('Shoreline Compactness vs. SAR Classification Performance');
  legend('Location','southeast');
end

function [xPct, y] = loadPairsAreaMetric(areaFile, metricFile)
  % Lê área (date, areaLago, value) e métrica (date, value), trata duplicatas e alinha por data
  Ta = readtable(areaFile);
  Tm = readtable(metricFile);

  Ta.date = datetime(Ta.date,'InputFormat','yyyy-MM-dd');
  Tm.date = datetime(Tm.date,'InputFormat','yyyy-MM-dd');

  TTa = table2timetable(Ta,'RowTimes','date');  % ('date','areaLago','value') — usar 'value'
  TTm = table2timetable(Tm,'RowTimes','date');  % ('date','value')

  % Remove duplicatas por média (área)
  TTa = sortrows(TTa);
  [ua,~,ia] = unique(TTa.date);
  if numel(ua) < height(TTa)
    TTa = timetable(ua, ...
      accumarray(ia, TTa.value,    [], @mean), ...
      accumarray(ia, TTa.areaLago, [], @mean), ...
      'VariableNames', {'value','areaLago'});
  end

  % Remove duplicatas por média (métrica)
  TTm = sortrows(TTm);
  [um,~,im] = unique(TTm.date);
  if numel(um) < height(TTm)
    TTm = timetable(um, accumarray(im, TTm.value, [], @mean), 'VariableNames', {'value'});
  end

  % Percentual de área (relativo ao máximo próprio) — eixo X
  areaVal = TTa.value;
  areaMax = max(areaVal,[],'omitnan');
  xPct = 100 * areaVal / max(areaMax, eps);    % [% de 0 a 100]

  % Alinhar métrica às datas da área (interp. linear) — eixo Y
  TTm_onA = retime(TTm, TTa.Properties.RowTimes, 'linear');
  y = TTm_onA.value;                            % métrica bruta (sem normalização)
end
