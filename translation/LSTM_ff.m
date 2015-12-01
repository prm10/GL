function [reconstruct,predict,errorR,errorP]=LSTM_ff(input0,label0,args)
errorR=0;
errorP=0;
for i2=1:length(input0)
    input=input0{i2};
    label=label0{i2};
    %% 前向传播
    % encoder
    inputEncoder=input;%t=1：T
    for i1=1:length(args.encoderLayer)-2
        [~,~,~,~,~,~,y1{i1}]=LSTM_step_ff(inputEncoder,args.WeightEncoder{i1});
        inputEncoder=y1{i1};
    end
    %最后一层
    w_k1=args.WeightEncoder{end}.w_k;
    b_k1=args.WeightEncoder{end}.b_k;
    z_k1=y1{end}(end,:)*w_k1+b_k1;%只记录最后一层的最后一个
    C1=tanh(z_k1);
    for i1=1:length(args.decoderLayer)-2
        C2{i1}=C1*args.TranR{i1}.w_k+args.TranR{i1}.b_k;
    end
    for i1=1:length(args.predictLayer)-2
        C3{i1}=C1*args.TranP{i1}.w_k+args.TranP{i1}.b_k;
    end
    
    % decoder
    inputDecoder(1,:)=[C1,zeros(1,size(input,2))];%t=1
    w_k2=args.WeightDecoder{end}.w_k;
    b_k2=args.WeightDecoder{end}.b_k;
    for t=1:size(input,1)
        inputR=inputDecoder(t,:);
        for lay_i=1:length(args.decoderLayer)-2
            if t==1
                inputY=zeros(1,args.decoderLayer(lay_i+1));
                inputC=C2{lay_i};%zeros(1,args.decoderLayer(lay_i+1));
            else
                inputY=y2{lay_i}(t-1,:);
                inputC=c2{lay_i}(t-1,:);
            end
            [~,~,~,~,c2{lay_i}(t,:),~,y2{lay_i}(t,:)]=LSTM_step_ff1(inputR,inputY,inputC,args.WeightDecoder{lay_i});
            inputR=y2{lay_i}(t,:);
        end
        %最后一层
        z_k2=inputR*w_k2+b_k2;
        inputDecoder(t+1,:)=[C1,tanh(z_k2)];
    end
    reconstruct=inputDecoder(2:end,size(C1,2)+1:end);
    
    % predict
    inputPredict(1,:)=[C1,zeros(1,size(label,2))];%t=1
    w_k3=args.WeightPredict{end}.w_k;
    b_k3=args.WeightPredict{end}.b_k;
    for t=1:size(label,1)
        inputP=inputPredict(t,:);
        for lay_i=1:length(args.predictLayer)-2
            if t==1
                inputY=zeros(1,args.predictLayer(lay_i+1));
                inputC=C3{lay_i};%zeros(1,args.predictLayer(lay_i+1));
            else
                inputY=y3{lay_i}(t-1,:);
                inputC=c3{lay_i}(t-1,:);
            end
            [~,~,~,~,c3{lay_i}(t,:),~,y3{lay_i}(t,:)]=LSTM_step_ff1(inputP,inputY,inputC,args.WeightPredict{lay_i});
            inputP=y3{lay_i}(t,:);
        end
        %最后一层
        z_k3=inputP*w_k3+b_k3;
        inputPredict(t+1,:)=[C1,tanh(z_k3)];
    end
    predict=inputPredict(2:end,size(C1,2)+1:end);

    % 计算误差
    errorR=errorR+sum(sum((input(end:-1:1,:)-reconstruct).^2))/size(reconstruct,2);
    errorP=errorP+sum(sum((label-predict).^2))/size(predict,2);
end
errorP=errorP/length(input0);
errorR=errorR/length(input0);