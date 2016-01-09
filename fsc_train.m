function [args]=fsc_train(args)
    global train_data train_label test_data test_label;
    tic;
    h=waitbar(0);
    for i1=1:args.maxecho
        waitbar(0,h,strcat('第',num2str(i1),'/',num2str(args.maxecho),'次迭代'));
        for i2=1:args.circletimes
            index=1+floor(rand(args.batchsize,1)*length(train_data));
            adw=cell(args.batchsize,1);
            for i3=1:args.batchsize%对单个样本计算梯度
                input=train_data{index(i3)};
                label=train_label{index(i3)};
                [adw{i3}]=fsc_ff_bp(args,input,label);
            end
            args=fsc_weight_update(args,adw);
            waitbar(i2/args.circletimes,h,strcat('第',num2str(i1),'/',num2str(args.maxecho),'次迭代：',num2str(i2),'/',num2str(args.circletimes)));
        end
       %% 统计误差
        [~,error]=fsc_ff(test_data,test_label,args);
        fprintf('echo: %d \t test error: %.4f\n',i1,error);
        args.Er=[args.Er;error];
    end
    close(h);
    toc;

