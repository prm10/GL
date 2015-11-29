function [args]=LSTM_train(args)
    global train_data train_label test_data test_label;
    h=waitbar(0);
    for i1=1:args.maxecho
        waitbar(0,h,strcat('��',num2str(i1),'/',num2str(args.maxecho),'�ε���'));
        index=1+floor(rand(args.circletimes,1)*length(train_data));
        for i2=1:args.circletimes%�Ե��������ݶ��½�
            input=train_data{index};
            label=train_label{index};
            [args]=LSTM_ff_bp(args,input,label);
            waitbar(i2/args.circletimes,h,strcat('��',num2str(i1),'/',num2str(args.maxecho),'�ε�����',num2str(i2),'/',num2str(args.circletimes)));
        end
       %% ͳ�����
        [~,~,errorR,errorP]=LSTM_ff(test_data,test_label,args);
        fprintf('echo: %d\ttest error: %.4f\treconstruct error: %.4f\tpredict error: %.4f\n',i1,errorR+errorP,errorR,errorP);
    end
    close(h);


