clc;close all;clear all;
rng(3);
load('K:\GL_data\5\data_labeled.mat');
train_input=input0([1:4]);
train_label=label0([1:4]);
test_input=input0([5,6]);
test_label=label0([5,6]);
% load('K:\GL_data\5\data_labeled.mat');
% test_input=input0;
% test_label=label0;
clear input0 label0;
args.layer=[size(train_input{1},2) 10 size(train_label{1},2)];
args.maxecho=100;
args.momentum=0.9;
args.learningrate=1e-3;
args.outputtype='softmax';
args=LSTM_initial(args);
[args]=LSTM_train(args,train_input,train_label);
for i1=1:length(test_label)
    [dout,error]=LSTM_ff(test_input(i1),test_label(i1),args);
    figure;
    area(dout);
    legend('1','2','3','4','5');
    title(num2str(find(test_label{i1}(end,:)==1)));
end

