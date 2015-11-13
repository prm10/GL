clc;close all;clear all;
rand('seed',2);
args.input=rand(5,3);
args.label=[mean(args.input,2)<=0.5 mean(args.input,2)>0.5];
args.numblocks=5;
args=LSTM_initial(args);
delta=1e-8;

% post=10;
% pos=2;
% [~,d1]=LSTM_ff(args,0,post,pos);
% [err1,~]=LSTM_ff(args,delta,post,pos);
% [err2,~]=LSTM_ff(args,-delta,post,pos);
% d2=(err1-err2)/2/delta
% d1(post,pos)

for post=1:5
    for pos=1:5
        [~,d1]=LSTM_ff(args,0,post,pos);
        % args.Weight.w_i(1,1)=args.Weight.w_i(1,1)+delta;
        [err1,~]=LSTM_ff(args,delta,post,pos);
        % args.Weight.w_i(1,1)=args.Weight.w_i(1,1)-2*delta;
        [err2,~]=LSTM_ff(args,-delta,post,pos);
        d2(post,pos)=(err1-err2)/2/delta;
    end
end



% 
% [~,~,dw_i]=LSTM_ff(args);
% args.Weight.w_i(2,1)=args.Weight.w_i(2,1)+1e-10;
% [err1]=LSTM_ff(args);
% args.Weight.w_i(2,1)=args.Weight.w_i(2,1)-2e-10;
% [err2]=LSTM_ff(args);
% (err1-err2)/2e-10/dw_i(2,1)