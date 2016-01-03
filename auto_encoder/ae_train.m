function [args]=ae_train(args,layer_i)
    global train_data test_data;
    tic;    
    h=waitbar(0);
    [batchSize,numDim,batches]=size(train_data);
    for i1=1:args.maxecho
%         waitbar(0,h,strcat('��',num2str(i1),'/',num2str(args.maxecho),'�ε���'));
        for i2=1:batches
            [args,error]=ae_gradient(args,layer_i,train_data(:,:,batches));
%             waitbar(i2/batches,h,strcat('��',num2str(i1),'/',num2str(args.maxecho),'�ε�����',num2str(i2),'/',num2str(batches)));
        end
       %% ͳ�����
%         [~,error]=ae_ff(test_data,args);
        if mod(i1,args.printEvery)==0
            fprintf('echo: %d\t error: %.4f\n',i1,1e4*error);
        end
    end
%     close(h);
%     delete(poolobj);
    toc;