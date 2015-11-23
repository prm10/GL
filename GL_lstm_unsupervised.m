clc;close all;clear all;
rng(3);
% load('K:\GL_data\3\data.mat');
load('K:\GL_data\3\data_normalized.mat');

% range=400000:length(date0);
% plot(date0(range),data0(range,6));
% datestr(date0(1920468),'yyyy-mm-dd HH:MM:SS')
%% 3�Ÿ�¯��2012-07-06�����Ϊѵ������2013��Ϊ���Լ�
global train_data train_label test_data test_label;
train_data=data1(400000:1920468,:);
test_data=data1(1920468:end,:);
clear data1;
args.layer=[size(train_input{1},2) 20 5 size(train_label{1},2)];
args.maxecho=100;
args.momentum=0.9;
args.labellength=8000;
args.circletimes=100;
args.learningrate=1e-3;
args.outputtype='softmax';
args=LSTM_initial(args);
[args]=LSTM_train(args,train_input,train_label);