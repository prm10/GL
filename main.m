clc;close all;clear;
No=[2,3,5];
GL=[7,1,5];
i1=1;
load(strcat('data\',num2str(No(i1)),'\data_labeled.mat'));

global train_data train_label test_data test_label;
train_data=[train_data;x1(range1,:)];
train_label=[train_label;x1(range2,:)];
test_data=train_data;
test_label=train_label;
load('args.mat');
if(exist('args','var'))%梯度检查
    args.maxecho=1;
    args.circletimes=1;
    args.momentum=0;
    args.learningrate=0;
    [args]=LSTM_train(args);%计算下当前的梯度
    vname='args.WeightTranP{1, 1}.w_k';
    vname=vname(6:end);
    s1=strcat('dcal=args.Mom.',vname,'(1,1);');
    eval(s1);
%     dcal=args.Mom.WeightPredict{1, 1}.w_i(1,1);
    error_delta=1e-10/dcal;%1e-5;%
    s2=strcat('args.',vname,'(1,1)=args.',vname,'(1,1)+error_delta;');
    eval(s2);
%     args.WeightPredict{1, 1}.w_i(1,1)=args.WeightPredict{1, 1}.w_i(1,1)+error_delta;
    [~,~,errorR1,errorP1]=LSTM_ff(train_data,train_label,args);
    s3=strcat('args.',vname,'(1,1)=args.',vname,'(1,1)-2*error_delta;');
    eval(s3);
%     args.WeightPredict{1, 1}.w_i(1,1)=args.WeightPredict{1, 1}.w_i(1,1)-2*error_delta;
    [~,~,errorR2,errorP2]=LSTM_ff(train_data,train_label,args);
    dreal=(errorR1+errorP1-errorR2-errorP2)/error_delta/2;
    accuracy=abs((dcal-dreal)/dreal)*100;
else %初始化
    args.maxecho=400;
    args.circletimes=10;
    args.momentum=0.9;
    args.learningrate=1e-1;
    args.batchsize=10;
    args.LengthDecoder=10;
    args.LengthPredict=10;
    dimC=10;
    dimInput=1;
    dimOutput=1;
    layer=[50];
    args.encoderLayer=[1,layer,dimC];
    args.decoderLayer=[dimC+dimInput,layer,dimInput];
    args.predictLayer=[dimC+dimOutput,layer,dimOutput];
    args=LSTM_initial(args);
    [args]=LSTM_train(args);
    save('args.mat','args');
%     test_data=train_data;
%     test_label=train_label;
%     [reconstruct,predict,errorR,errorP]=LSTM_ff(test_data,test_label,args);
end

