function [args]=ae_train(args)
    global train_data train_label test_data test_label;
%     poolobj = parpool();
    tic;
    h=waitbar(0);
    for i1=1:args.maxecho
        waitbar(0,h,strcat('第',num2str(i1),'/',num2str(args.maxecho),'轮迭代：'));
        for i2=1:args.circletimes
            index=1+floor(rand(args.batchsize,1)*length(train_data));
            adw=cell(args.batchsize,1);
            train_data1=train_data(index);
            train_label1=train_label(index);
            for i3=1:args.batchsize
                input=train_data1{i3};
                label=train_label1{i3};
                adw{i3}=ae_ff_bp(args,input,label);
            end
            args=ae_weight_update(args,adw);
            waitbar(i2/args.circletimes,h,strcat('第',num2str(i1),'/',num2str(args.maxecho),'轮迭代：',num2str(i2),'/',num2str(args.circletimes)));
        end
       
        [~,error]=ae_ff(test_data,test_label,args);
        fprintf('echo: %d \t test error: %.4f\n',i1,error);
        args.Er=[args.Er;error];
    end
    close(h);
    toc;
%     delete(poolobj);
