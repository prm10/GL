function [reconstruct,predict,errorR,errorP]=LSTM_ff(input0,label0,args)
errorR=0;
errorP=0;
for i2=1:length(input0)
    input=input0{i2};
    label=label0{i2};
    %% 前向传播
    % encoder
    input_x11=input(1,:);%t=1
    [~,~,~,~,~,~,~,C]=LSTM_step_ff1(input_x11,0,input,args.WeightEncoder,size(input,1));
    C1=C(end,:);
    %初始状态层
    for i1=1:length(args.decoderLayer)-2
        C2{i1}=C1*args.WeightTranR{i1}.w_k+args.WeightTranR{i1}.b_k;
    end
    for i1=1:length(args.predictLayer)-2
        C3{i1}=C1*args.WeightTranP{i1}.w_k+args.WeightTranP{i1}.b_k;
    end
    % decoder
    input_x11=[C1,zeros(1,size(input,2))];%t=1
    [~,~,~,~,~,~,~,reconstruct]=LSTM_step_ff1(input_x11,C2,C1,args.WeightDecoder,size(input,1));
    % predict
    input_x11=[C1,zeros(1,size(label,2))];%t=1
    [~,~,~,~,~,~,~,predict]=LSTM_step_ff1(input_x11,C3,C1,args.WeightPredict,size(label,1));
    % 计算误差
    errorR=errorR+sum(sum((input(end:-1:1,:)-reconstruct).^2))/size(reconstruct,2);
    errorP=errorP+sum(sum((label-predict).^2))/size(predict,2);
end
errorP=errorP/length(input0)/2;
errorR=errorR/length(input0)/2;