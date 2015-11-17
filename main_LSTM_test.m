clc;close all;clear all;
rand('seed',3);
%% ���룺��ά�������
% �����1����1ά�����ǰ3��ͣ�2����2ά��ȥ��1ά��3��
for i1=1:100
    input{i1}=rand(100,3);
    label{i1}=[mean(input{i1},2)<=0.5 mean(input{i1},2)>0.5];
end

args.layer=[size(input{1},2) 10 size(label{1},2)];
args.maxecho=200;
args.learningrate=1e-1;
args=LSTM_initial(args);
[args]=LSTM_train(args,input,label);

% test set
test_input{1}=rand(10000,3);
test_label{1}=[mean(test_input{1},2)<=0.5 mean(test_input{1},2)>0.5];
[dout,error]=LSTM_ff(test_input,test_label,args);
