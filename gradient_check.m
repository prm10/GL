clc;close all;clear all;
rand('seed',3);
args.input=rand(10,3);
args.label=[mean(args.input,2)<=0.5 mean(args.input,2)>0.5];
args.numblocks=5;
args=LSTM_initial(args);
[~,~,dw_i]=LSTM_ff(args);
args.Weight.w_i(1,1)=args.Weight.w_i(1,1)+1e-8;
[err1]=LSTM_ff(args);
args.Weight.w_i(1,1)=args.Weight.w_i(1,1)-2e-8;
[err2]=LSTM_ff(args);
(err1-err2)/2e-8
dw_i(1,1)