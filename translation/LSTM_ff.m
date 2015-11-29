function [reconstruct,predict,errorR,errorP]=LSTM_ff(input0,label0,args)
errorR=0;
errorP=0;
for i2=1:length(input0)
    input=input0{i2};
    label=label0{i2};
    y1=input;
    %% 前向传播
    % encoder
    for i1=1:length(args.encoderLayer)-2
        [~,~,~,~,~,~,y1]=LSTM_step_ff(y1,args.WeightEncoder{i1});
    end
    %最后一层
    w_k1=args.WeightEncoder{end}.w_k;
    b_k1=args.WeightEncoder{end}.b_k;
    z_k1=y1(end,:)*w_k1+b_k1;%只记录最后一层的最后一个
    C=tanh(z_k1);
    % decoder
    inputDecoder(1,:)=[C,zeros(1,size(input,2))];
    for i3=1:size(input,1)
        inputR=inputDecoder(i3,:);
        for i1=1:length(args.decoderLayer)-2
            [~,~,~,~,~,~,inputR]=LSTM_step_ff(inputR,args.WeightDecoder{i1});
        end
        %最后一层
        w_k2=args.WeightDecoder{end}.w_k;
        b_k2=args.WeightDecoder{end}.b_k;
        z_k2=inputR*w_k2+ones(size(inputR,1),1)*b_k2;
        inputDecoder(i3+1,:)=[C,tanh(z_k2)];
    end
    reconstruct=inputDecoder(2:end,size(C,2)+1:end);
    % predict
    inputPredict(1,:)=[C,zeros(1,size(label,2))];
    for i3=1:size(label,1)
        inputP=inputPredict(i3,:);
        for i1=1:length(args.predictLayer)-2
            [~,~,~,~,~,~,inputP]=LSTM_step_ff(inputP,args.WeightPredict{i1});
        end
        %最后一层
        w_k3=args.WeightPredict{end}.w_k;
        b_k3=args.WeightPredict{end}.b_k;
        z_k3=inputP*w_k3+ones(size(inputP,1),1)*b_k3;
        inputPredict(i3+1,:)=[C,tanh(z_k3)];
    end
    predict=inputPredict(2:end,size(C,2)+1:end);

    % 计算误差
    errorR=errorR+sum(sum((input(end:-1:1,:)-reconstruct).^2))/size(reconstruct,1)/size(reconstruct,2);
    errorP=errorP+sum(sum((label-predict).^2))/size(predict,1)/size(predict,2);
end
errorP=errorP/length(input0)*100;
errorR=errorR/length(input0)*100;