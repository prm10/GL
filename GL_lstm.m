clc;close all;clear all;
rng(3);
load('K:\GL_data\3\data_labeled.mat');
train_index=[1:5,10:15];
test_index=[6:9,15:19];
train_input=input0(train_index);
train_label=label0(train_index);
test_input=input0(test_index);
test_label=label0(test_index);
% load('K:\GL_data\5\data_labeled.mat');
% test_input=input0;
% test_label=label0;
clear input0 label0;
args.layer=[size(train_input{1},2) 20 5 size(train_label{1},2)];
args.maxecho=100;
args.momentum=0.9;
args.learningrate=1e-3;
args.outputtype='softmax';
args=LSTM_initial(args);
[args]=LSTM_train(args,train_input,train_label);
for i1=1:length(test_label)
    [dout,error]=LSTM_ff(test_input(i1),test_label(i1),args);
    figure;
    area(dout{1});
    legend('1','2','3','4','5');
    title(num2str(find(test_label{i1}(end,:)==1)));
end

