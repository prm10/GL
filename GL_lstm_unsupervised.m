clc;close all;clear all;
rng(3);
% load('K:\GL_data\3\data.mat');
load('K:\GL_data\3\data_normalized.mat');

% range=400000:length(date0);
% plot(date0(range),data0(range,6));
% datestr(date0(1920468),'yyyy-mm-dd HH:MM:SS')
%% 3号高炉，2012-07-06至年底为训练集；2013年为测试集
train_data=data1(400000:1920468,:);
test_data=data1(1920468:end,:);
clear data1;
