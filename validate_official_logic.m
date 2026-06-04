clear; clc; close all;

addpath(genpath('/public/home/achwjznh4b/install/toolbox/mexcdf/'));

ref_path = getenv('MCC_REF_CLIM_PATH');
if isempty(ref_path)
    ref_path = '/public/home/achwjznh4b/ERA5/Climatology/';
end
contestant_clim_path = getenv('MCC_CONTESTANT_CLIM_PATH');
if isempty(contestant_clim_path)
    contestant_clim_path = '/public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601/output/';
end
save_path = getenv('MCC_VERIFY_SAVE_PATH');
if isempty(save_path)
    save_path = '/public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601/verification/';
end

if ref_path(end) ~= '/', ref_path = [ref_path, '/']; end
if contestant_clim_path(end) ~= '/', contestant_clim_path = [contestant_clim_path, '/']; end
if save_path(end) ~= '/', save_path = [save_path, '/']; end
mkdir(save_path);

files_env = getenv('MCC_VALIDATE_FILES');
if isempty(files_env)
    file_list = {};
    days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31];
    for month = 6:8
        for day = 1:days_in_month(month)
            file_list{end + 1} = sprintf('%02d%02d.nc', month, day); %#ok<SAGROW>
        end
    end
else
    raw = strsplit(files_env, ',');
    file_list = {};
    for i = 1:numel(raw)
        item = strtrim(raw{i});
        if isempty(item), continue; end
        if length(item) == 4
            item = [item, '.nc'];
        end
        file_list{end + 1} = item; %#ok<SAGROW>
    end
end

rmse_clim = nan(1, numel(file_list));
rmse_P90 = nan(1, numel(file_list));

fprintf('=========================================\n');
fprintf('       MCC official-logic verification\n');
fprintf('=========================================\n');
fprintf('Reference path  : %s\n', ref_path);
fprintf('Contestant path : %s\n', contestant_clim_path);
fprintf('Save path       : %s\n', save_path);
fprintf('File count      : %d\n', numel(file_list));
fprintf('=========================================\n');

for day_index = 1:numel(file_list)
    filename = file_list{day_index};
    fprintf('正在读取：%s\n', filename);

    clim_file = [ref_path, filename];
    clim_data = ncread(clim_file, 'Climmean');
    P90_data = ncread(clim_file, 'P90_sst');

    contestant_file = [contestant_clim_path, filename];
    contestant_data = ncread(contestant_file, 'Climmean');
    contestant_P90 = ncread(contestant_file, 'P90_sst');

    rmse_clim(day_index) = sqrt(nanmean((clim_data(:) - contestant_data(:)).^2));
    rmse_P90(day_index) = sqrt(nanmean((P90_data(:) - contestant_P90(:)).^2));
    fprintf('  Clim RMSE=%.6f, P90 RMSE=%.6f\n', rmse_clim(day_index), rmse_P90(day_index));
end

R_clim = nanmean(rmse_clim);
R_P90 = nanmean(rmse_P90);
max_clim = max(rmse_clim);
max_P90 = max(rmse_P90);

fprintf('=========================================\n');
fprintf('       6~8月夏季 RMSE 统计结果\n');
fprintf('=========================================\n');
fprintf('气候态 平均误差 : %.4f ℃\n', R_clim);
fprintf('气候态 最大误差 : %.4f ℃\n', max_clim);
fprintf('-----------------------------------------\n');
fprintf('P90分位 平均误差 : %.4f ℃\n', R_P90);
fprintf('P90分位 最大误差 : %.4f ℃\n', max_P90);
fprintf('=========================================\n');

dlmwrite(fullfile(save_path, 'RMSE_clim.txt'), R_clim, 'precision', '%.4f');
dlmwrite(fullfile(save_path, 'RMSE_P90.txt'), R_P90, 'precision', '%.4f');
dlmwrite(fullfile(save_path, 'RMSE_clim_daily.txt'), rmse_clim(:), 'precision', '%.6f');
dlmwrite(fullfile(save_path, 'RMSE_P90_daily.txt'), rmse_P90(:), 'precision', '%.6f');
