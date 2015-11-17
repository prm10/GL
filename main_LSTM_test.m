clc;close all;clear all;
rand('seed',4);
%% ���룺��ά�������
% �����1����1ά�����ǰ3��x(t-1)�ĺͣ�2����2ά��ȥ��1ά��
for i1=1:10
    input{i1}=rand(100,3);
%     label{i1}=[mean(input{i1},2)<=0.5 mean(input{i1},2)>0.5];
    l1=[0;0;sum(input{i1}(1:end-2,:),2)];
    l2=[-input{i1}(1,2);(input{i1}(1:end-1,1)-input{i1}(2:end,2))];
    l3=[zeros(8,1);input{i1}(1:end-8,3)];
    label{i1}=[l1 l2 l3];
end

args.layer=[size(input{1},2) 10 size(label{1},2)];
args.maxecho=20;
args.momentum=0.9;
args.learningrate=1e-1;
args.outputtype='softmax';
args=LSTM_initial(args);
[args]=LSTM_train(args,input,label);

% test set
test_input{1}=rand(10000,3);
test_label{1}=[mean(test_input{1},2)<=0.5 mean(test_input{1},2)>0.5];
[dout,error]=LSTM_ff(test_input,test_label,args);
