function [args]=LSTM_train(args,input,label)
    for i1=1:args.maxecho
        for i2=1:length(input)%�Ե��������ݶ��½�
            [args]=LSTM_ff_bp(args,input{i2},label{i2});
        end
       %% ͳ�����
        [~,error]=LSTM_ff(input,label,args);
        fprintf('%d train error: %.4f \n',i1,error);
    end


