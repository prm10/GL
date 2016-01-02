function [args]=ae_train(args,layer_i)
    global train_data train_label test_data test_label;
    tic;    
    h=waitbar(0);
    batches=floor(size(train_data,1)/args.batchsize);
    args.batches=batches;
    for i1=1:args.maxecho
        waitbar(0,h,strcat('第',num2str(i1),'/',num2str(args.maxecho),'次迭代'));
        for i2=1:batches
            index=args.batchsize*(batches-1)+1:args.batchsize*batches;
            [args]=ae_gradient(args,layer_i,train_data(index,:),train_label(index,:));
%             waitbar(i2/batches,h,strcat('第',num2str(i1),'/',num2str(args.maxecho),'次迭代：',num2str(i2),'/',num2str(batches)));
        end
       %% 统计误差
        [~,~,errorR,errorP]=ae_ff(test_data,test_label,args);
        fprintf('echo: %d\ttest error: %.4f\treconstruct error: %.4f\tpredict error: %.4f\n',i1,1e4*(errorR+errorP),1e4*errorR,1e4*errorP);
    end
    close(h);
%     delete(poolobj);
    toc;