function [predict,error]=fsc_ff(input0,label0,args)
error=0;
for i2=1:length(input0)
    input=input0{i2};
    label=label0{i2};
    %% 前向传播
    % encoder
    input_x11=input(1,:);%t=1
    [~,~,~,~,~,~,~,predict]=LSTM_step_ff1(input_x11,0,input,args.Weight,size(input,1),args.outputLayer);
    % 计算误差
    switch args.outputLayer
        case{'softmax'}
            error=error-sum(sum(label.*log(predict)))/size(predict,1)*2;
        otherwise
            error=error+sum(sum((label-predict).^2))/size(predict,1)/size(predict,2);
    end
end
error=error/length(input0)/2;