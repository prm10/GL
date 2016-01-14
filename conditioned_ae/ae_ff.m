function [predict,error]=fsc_ff(input0,label0,args)
error=0;
for i2=1:length(input0)
    input=input0{i2};
    label=label0{i2};
    %% 前向传播
    x1=input;
    for i1=1:length(args.layer)-2
        c0=zeros(1,size(args.Weight{i1}.r_i,1));
        [~,~,~,~,~,~,y2]=LSTM_step_ff_fast(x1,c0,args.Weight{i1});
        x1=y2;
    end
    predict=LSTM_output_ff(args.outputLayer,args.Weight{end}.w_k,args.Weight{end}.b_k,y2);
    
    % 计算误差
    switch args.outputLayer
        case{'softmax'}
            error=error-sum(sum(label.*log(predict)))/size(predict,1)*2;
        otherwise
            error=error+sum(sum((label-predict).^2))/size(predict,1)/size(predict,2);
    end
end
error=error/length(input0)/2;