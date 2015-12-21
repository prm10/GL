clc;close all;clear;
No=[2,3,5];
GL=[7,1,5];
ipt=[1;8;13;17;20;24];
plotvariable;
i1=1;
load(strcat('data\',num2str(No(i1)),'\data_labeled.mat'));
i2=4;%:length(input0)
data1=input0{i2}(:,commenDim{GL(i1)});


i0=17;
hotWindPress=data1(:,i0);
[indexChange]=FindStoveChange(hotWindPress);
% figure;
% plot(find(~indexChange),hotWindPress(~indexChange),'b.',find(indexChange),hotWindPress(indexChange),'r.');


% md=zeros(length(hotWindPress),1);
% sd=zeros(length(hotWindPress),1);
% for i1=1:length(hotWindPress)
%     index=(max(1,i1-720):i1);
%     tempData=hotWindPress(index);
%     md(i1)=median(tempData);
%     sd(i1)=std(tempData);
% end
% % flag=(coldWind-M(i0))./S(i0);
% flag=(hotWindPress-md)./sd;
% 
% figure;
% subplot(311);
% plot(md);
% subplot(312);
% plot(hotWindPress,'.');
% title(commenVar{i0});
% subplot(313);
% plot(flag);

% figure;%datestr(time0{2}(end),'yyyy-mm-dd HH:MM:SS')
% for i1=1:6
%     subplot(3,2,i1);
%     plot(data1(:,ipt(i1)));%1e4:end
%     title(commenVar{ipt(i1)});
% end


% [input,decode,predict]=GenerateData(data,lengthD,lengthP,indexD,indexP);


%% 
global train_data train_label test_data test_label;
train_data=cell(0);
train_label=cell(0);
rng(3);
lenInput=360;
lenOutput=1;
num=10000;
index=floor(rand(num,1)*(size(hotWindPress,1)-lenInput-lenOutput));
for i1=1:num
    range1=index(i1)+1:index(i1)+lenInput;
    range2=index(i1)+lenInput+1:index(i1)+lenInput+lenOutput;
    train_data=[train_data;hotWindPress(range1,:)];
    train_label=[train_label;hotWindPress(range2,:)];
end
% plot(train_data{10},'.')

test_data=cell(0);
test_label=cell(0);
num=10;
index=floor(rand(num,1)*(size(hotWindPress,1)-lenInput-lenOutput));
for i1=1:num
    range1=index(i1)+1:index(i1)+lenInput;
    range2=index(i1)+lenInput+1:index(i1)+lenInput+lenOutput;
    test_data=[test_data;hotWindPress(range1,:)];
    test_label=[test_label;hotWindPress(range2,:)];
end

% 参数设置
choice=2;
switch choice
    case 1%初始化
        args.maxecho=10;
        args.circletimes=10;
        args.momentum=0.9;
        args.learningrate=1e-1;
        args.batchsize=10;
        args.LengthDecoder=20;
        dimC=10;
        dimInput=1;
        dimOutput=1;
        layer=[100];
        args.encoderLayer=[dimInput,layer,dimC];
        args.decoderLayer=[dimC+dimInput,layer,dimInput];
        args.predictLayer=[dimC+dimOutput,layer,dimOutput];
        args=LSTM_initial(args);
        [args]=LSTM_train(args);
        save('args_hot_wind_pressure.mat','args');
    case 2%继续运算
        load('args_hot_wind_pressure.mat');
        args.maxecho=100;
        args.circletimes=100;
        args.momentum=0.99;
%         args.learningrate=1e-1;
        args.batchsize=10;
        [args]=LSTM_train(args);
        save('args_hot_wind_pressure.mat','args');
    case 3%梯度检验
        load('args_hot_wind_pressure.mat');
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
end

i1=3;
[reconstruct,predict,errorR,errorP]=LSTM_ff(test_data(i1),test_label(i1),args);
figure;
plot(1:size(test_data{i1},1),test_data{i1},'--o'...
    ,size(test_data{i1},1)+1:size(test_data{i1},1)+size(test_label{i1},1),test_label{i1},'--o'...
    ,size(test_data{i1},1):-1:size(test_data{i1},1)-size(reconstruct,1)+1,reconstruct,'*'...
    ,size(test_data{i1},1)+1:size(test_data{i1},1)+size(predict,1),predict,'*');

