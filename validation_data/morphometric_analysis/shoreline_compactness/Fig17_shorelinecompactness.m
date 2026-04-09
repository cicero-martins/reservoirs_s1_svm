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

plotUnifiedMetricRaw('AP', 'A/P', 'A/P (m)', ...
    metricFiles.AP, areaFiles, reservoirNames, colors, markers);

% plotUnifiedMetricRaw('DL', 'DL', 'DL (-)', ...
%     metricFiles.DL, areaFiles, reservoirNames, colors, markers);

%% ====================== FUNÇÕES LOCAIS ======================
function plotUnifiedMetricRaw(metricKey, metricTitle, yLabel, fileList, areaFiles, rnames, colors, markers)
  figure('Name', metricKey); hold on; grid on; box on;

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
