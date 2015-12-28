clc;close all;clear;

args_name='args_ae.mat';
choice=2;
switch choice
    case 1%初始化
        args.maxecho=10;
        args.circletimes=100;
        args.momentum=0.9;
        args.weightDecay=1e-5;
        args.learningrate=1e-1;
        args.batchsize=3;
        args.layer=[1,20,2];
        args.Er=[];
%         args.outputLayer='softmax';
        args=ae_initial(args);
        [args]=ae_train(args);
        save(args_name,'args');
    case 2%继续运算
        load(args_name);
        args.maxecho=100;
        args.circletimes=100;
%         args.momentum=0.5;
%         args.learningrate=5e-2;
        args.batchsize=3;
        [args]=fsc_train(args);
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

