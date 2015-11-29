function [args]=LSTM_ff_bp(args,input,label)
    %% 前向传播
    % encoder
    inputEncoder=input;
    for i1=1:length(args.encoderLayer)-2
        [x1{i1},in1{i1},f1{i1},z1{i1},c1{i1},o1{i1},y1{i1}]=LSTM_step_ff(inputEncoder,args.WeightEncoder{i1});
        inputEncoder=y1{i1};
    end
    %最后一层
    w_k1=args.WeightEncoder{end}.w_k;
    b_k1=args.WeightEncoder{end}.b_k;
    z_k1=y1{end}(end,:)*w_k1+b_k1;%只记录最后一层的最后一个
    C=tanh(z_k1);
    % decoder
    inputDecoder=[ones(size(input,1),1)*C,[zeros(1,size(input,2));input(end:-1:2,:)]];
    for i1=1:length(args.decoderLayer)-2
        [x2{i1},in2{i1},f2{i1},z2{i1},c2{i1},o2{i1},y2{i1}]=LSTM_step_ff(inputDecoder,args.WeightDecoder{i1});
        inputDecoder=y2{i1};
    end
     %最后一层
    w_k2=args.WeightDecoder{end}.w_k;
    b_k2=args.WeightDecoder{end}.b_k;
    z_k2=y2{end}*w_k2+ones(size(y2{end},1),1)*b_k2;
    reconstruct=tanh(z_k2);
    
    % predict
    inputPredict=[ones(size(label,1),1)*C,[zeros(1,size(label,2));label(1:end-1,:)]];
    for i1=1:length(args.predictLayer)-2
        [x3{i1},in3{i1},f3{i1},z3{i1},c3{i1},o3{i1},y3{i1}]=LSTM_step_ff(inputPredict,args.WeightPredict{i1});
        inputPredict=y3{i1};
    end
     %最后一层
    w_k3=args.WeightPredict{end}.w_k;
    b_k3=args.WeightPredict{end}.b_k;
    z_k3=y3{end}*w_k3+ones(size(y3{end},1),1)*b_k3;
    predict=tanh(z_k3);
    %% 反向传播
    % predict layer
    delta_k3=-(label-predict).*(1-predict.^2);
    dw_k3=y3{end}'*delta_k3/size(delta_k3,1);
    db_k3=mean(delta_k3,1);
    args.Mom.WeightPredict{end}.w_k=args.momentum*args.Mom.WeightPredict{end}.w_k+dw_k3;
    args.Mom.WeightPredict{end}.b_k=args.momentum*args.Mom.WeightPredict{end}.b_k+db_k3;
    delta_up3=delta_k3*w_k3';
    args.WeightPredict{end}.w_k=w_k3-args.learningrate*args.Mom.WeightPredict{end}.w_k;
    args.WeightPredict{end}.b_k=b_k3-args.learningrate*args.Mom.WeightPredict{end}.b_k;
    % lstm layer
    for i1=length(args.predictLayer)-2:-1:1
        [delta_up3,WeightPredict,MomWeightPredict]=LSTM_step_bp(args,delta_up3,args.WeightPredict{i1},args.Mom.WeightPredict{i1},x3{i1},in3{i1},f3{i1},z3{i1},c3{i1},o3{i1},y3{i1});
        args.WeightPredict{i1}=WeightPredict;
        args.Mom.WeightPredict{i1}=MomWeightPredict;
    end
    
    % reconstruction layer
    delta_k2=-(input(end:-1:1,:)-reconstruct).*(1-reconstruct.^2);
    dw_k2=y2{end}'*delta_k2/size(delta_k2,1);
    db_k2=mean(delta_k2,1);
    args.Mom.WeightDecoder{end}.w_k=args.momentum*args.Mom.WeightDecoder{end}.w_k+dw_k2;
    args.Mom.WeightDecoder{end}.b_k=args.momentum*args.Mom.WeightDecoder{end}.b_k+db_k2;
    delta_up2=delta_k2*w_k2';
    args.WeightDecoder{end}.w_k=w_k2-args.learningrate*args.Mom.WeightDecoder{end}.w_k;
    args.WeightDecoder{end}.b_k=b_k2-args.learningrate*args.Mom.WeightDecoder{end}.b_k;
    % lstm layer
    for i1=length(args.decoderLayer)-2:-1:1
        [delta_up2,WeightDecoder,MomWeightDecoder]=LSTM_step_bp(args,delta_up2,args.WeightDecoder{i1},args.Mom.WeightDecoder{i1},x2{i1},in2{i1},f2{i1},z2{i1},c2{i1},o2{i1},y2{i1});
        args.WeightDecoder{i1}=WeightDecoder;
        args.Mom.WeightDecoder{i1}=MomWeightDecoder;
    end
    
    %encoder layer
    delta_k1=sum(delta_up2(:,1:size(C,2)),1)+sum(delta_up3(:,1:size(C,2)),1);
    dw_k1=y1{end}(end,:)'*delta_k1;
    db_k1=delta_k1;
    args.Mom.WeightEncoder{end}.w_k=args.momentum*args.Mom.WeightEncoder{end}.w_k+dw_k1;
    args.Mom.WeightEncoder{end}.b_k=args.momentum*args.Mom.WeightEncoder{end}.b_k+db_k1;
    delta_up1=[zeros(size(input,1)-1,size(w_k1,1));delta_k1*w_k1'];
    args.WeightEncoder{end}.w_k=w_k1-args.learningrate*args.Mom.WeightEncoder{end}.w_k;
    args.WeightEncoder{end}.b_k=b_k1-args.learningrate*args.Mom.WeightEncoder{end}.b_k;
    % lstm layer
    for i1=length(args.encoderLayer)-2:-1:1
        [delta_up1,WeightEncoder,MomWeightEncoder]=LSTM_step_bp(args,delta_up1,args.WeightEncoder{i1},args.Mom.WeightEncoder{i1},x1{i1},in1{i1},f1{i1},z1{i1},c1{i1},o1{i1},y1{i1});
        args.WeightEncoder{i1}=WeightEncoder;
        args.Mom.WeightEncoder{i1}=MomWeightEncoder;
    end
    
    


