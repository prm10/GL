function [args]=LSTM_train(args)
    global train_data train_label test_data test_label;
    h=waitbar(0);
    for i1=1:args.maxecho
        waitbar(0,h,strcat('第',num2str(i1),'/',num2str(args.maxecho),'次迭代'));
        index=1+floor(rand(args.circletimes,1)*length(train_data));
        for i2=1:args.circletimes%对单个样本梯度下降
            input=train_data{index};
            label=train_label{index};
            [args]=LSTM_ff_bp(args,input,label);
            waitbar(i2/args.circletimes,h,strcat('第',num2str(i1),'/',num2str(args.maxecho),'次迭代：',num2str(i2),'/',num2str(args.circletimes)));
        end
       %% 统计误差
        [~,~,errorR,errorP]=LSTM_ff(test_data,test_label,args);
        fprintf('echo: %d\ttest error: %.4f\treconstruct error: %.4f\tpredict error: %.4f\n',i1,errorR+errorP,errorR,errorP);
    end
    close(h);


