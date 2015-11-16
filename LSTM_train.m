function [args]=LSTM_train(args,input,label)
    for i1=1:args.maxecho
        for i2=1:length(input)%对单个样本梯度下降
            [args]=LSTM_ff_bp(args,input{i2},label{i2});
        end
       %% 统计误差
        [~,error]=LSTM_ff(input,label,args);
        fprintf('%d train error: %.4f \n',i1,error);
    end


