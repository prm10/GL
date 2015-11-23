function [args]=LSTM_train(args)
    global train_data train_label test_data test_label;
    n=size(train_data)-args.labellength;
    for i1=1:args.maxecho
        index=floor(rand(args.circletimes,1)*n);
        for i2=1:args.circletimes%�Ե��������ݶ��½�
            range=index(i2)+1:index(i2)+args.labellength;
            input=train_data(range,:);
            label=train_label(range,:);
            [args]=LSTM_ff_bp(args,input,label);
        end
       %% ͳ�����
        [~,error]=LSTM_ff(test_data,test_label,args);
        fprintf('%d train error: %.4f \n',i1,error);
    end


