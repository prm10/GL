function [args]=LSTM_train(args)
    global train_data train_label test_data test_label;
    h=waitbar(0);
    for i1=1:args.maxecho
        adw=cell(args.circletimes,1);
        waitbar(0,h,strcat('��',num2str(i1),'/',num2str(args.maxecho),'�ε���'));
        for i2=1:args.circletimes
            index=1+floor(rand(args.circletimes,1)*length(train_data));
            for i3=1:args.batchsize%�Ե��������ݶ��½�
                input=train_data{index};
                label=train_label{index};
                [adw{i3}]=LSTM_ff_bp(args,input,label);
            end
            args=LSTM_weight_update(args,adw);
            waitbar(i3/args.circletimes,h,strcat('��',num2str(i1),'/',num2str(args.maxecho),'�ε�����',num2str(i2),'/',num2str(args.circletimes)));
        end
       %% ͳ�����
        [~,~,errorR,errorP]=LSTM_ff(test_data,test_label,args);
        fprintf('echo: %d\ttest error: %.4f\treconstruct error: %.4f\tpredict error: %.4f\n',i1,100*(errorR+errorP),100*errorR,100*errorP);
    end
    close(h);


