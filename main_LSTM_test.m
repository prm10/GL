clc;close all;clear all;
rand('seed',2);
input{1}=rand(15,3);
label{1}=[mean(input{1},2)<=0.5 mean(input{1},2)>0.5];
args.layer=[size(input{1},2) 3 size(label{1},2)];
args.maxecho=1000;
args.learningrate=1e-4;
args=LSTM_initial(args);
args=LSTM_train(args,input,label);
