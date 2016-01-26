clc;close all;clear;
%
No=[2,3,5];
GL=[7,1,5];
ipt=[1;8;13;17;20;24];
plotvariable;
i1=2;%高炉编号
load(strcat('..\data\',num2str(No(i1)),'\data_labeled.mat'));
i2=1;%:length(input0)
data1=input0{i2}(:,commenDim{GL(i1)});
i3=17;%变量类型
hotWindPress=data1(:,i3);
global train_data test_data;
train_len=6e3;
batchSize=20;
timeStep=6;
timeLen=360;
train_data=generateBatches(hotWindPress(1:train_len,:),batchSize,timeStep,timeLen);
test_data=generateBatches(hotWindPress(train_len+1:end,:),1,timeStep,timeLen);

args_name='args_ae.mat';
choice=2;
switch choice
    case 1%初始化
        args.maxecho=1e4;
        args.momentum=0.9;
%         args.weightDecay=1e-5;
        args.learningrate=1e-2;
        args.printEvery=100;
        args.Er=[];
        args.lambda=0;
        args.layers=[size(train_data,2),10,size(train_data,2)];
%         args.outputLayer='softmax';
        args=ae_initial(args);
        [args]=ae_train(args,1);
        save(args_name,'args');
    case 2%继续运算
        load(args_name);
        args.maxecho=100;
%         args.circletimes=100;
%         args.momentum=0.5;
%         args.learningrate=5e-2;
%         args.batchsize=3;
        [args]=ae_train(args,1);
        save(args_name,'args');
    case 3%梯度检验
        load(args_name);
        args.maxecho=1;
        args.circletimes=1;
        args.momentum=0;
        args.learningrate=0;
        [args]=fsc_train(args);%计算下当前的梯度
        vname='args.Weight{1, 1}.w_i';
        vname=vname(6:end);
        s1=strcat('dcal=args.Mom.',vname,'(1,1);');
        eval(s1);
    %     dcal=args.Mom.WeightPredict{1, 1}.w_i(1,1);
        error_delta=1e-10/dcal;%1e-5;%
        s2=strcat('args.',vname,'(1,1)=args.',vname,'(1,1)+error_delta;');
        eval(s2);
    %     args.WeightPredict{1, 1}.w_i(1,1)=args.WeightPredict{1, 1}.w_i(1,1)+error_delta;
        [~,error1]=fsc_ff(train_data,train_label,args);
        s3=strcat('args.',vname,'(1,1)=args.',vname,'(1,1)-2*error_delta;');
        eval(s3);
    %     args.WeightPredict{1, 1}.w_i(1,1)=args.WeightPredict{1, 1}.w_i(1,1)-2*error_delta;
        [~,error2]=fsc_ff(train_data,train_label,args);
        dreal=(error1-error2)/error_delta/2;
        accuracy=abs((dcal-dreal)/dreal)*100;
end

[dout,error]=ae_ff(train_data(:,:,1),args,1);
plot(1:timeLen,train_data(1,:,1),1:timeLen,dout(1,:));

