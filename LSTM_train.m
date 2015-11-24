function [args]=LSTM_train(args)
    global train_data train_label test_data test_label;
    n=size(train_data)-args.labellength;
    h=waitbar(0);
    for i1=1:args.maxecho
        waitbar(0,h,strcat('��',num2str(i1),'/',num2str(args.maxecho),'�ε���'));
        index=floor(rand(args.circletimes,1)*n);
        for i2=1:args.circletimes%�Ե��������ݶ��½�
            range=index(i2)+1:index(i2)+args.labellength;
            input=train_data(range,:);
            label=train_label(range,:);
            [args]=LSTM_ff_bp(args,input,label);
            waitbar(i2/args.circletimes,h,strcat('��',num2str(i1),'/',num2str(args.maxecho),'�ε�����',num2str(i2),'/',num2str(args.circletimes)));
        end
       %% ͳ�����
        index=floor(rand(10,1)*(size(test_data)-args.labellength));
        testd1=cell(0);
        testl1=cell(0);
        for i2=1:10
            range=index(i2)+1:index(i2)+args.labellength;
            input=test_data(range,:);
            label=test_label(range,:);
            testd1=[testd1;input];
            testl1=[testl1;label];
        end
        [~,error]=LSTM_ff(testd1,testl1,args);
        fprintf('%d train error: %.4f \n',i1,error);
    end
    close(h);


