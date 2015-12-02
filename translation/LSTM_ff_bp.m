function [args]=LSTM_ff_bp(args,input,label)
    %% 前向传播
    % encoder
    input_x11=input(1,:);%t=1
    [x1,in1,f1,z1,c1,o1,y1,C]=LSTM_step_ff1(input_x11,0,input,args.WeightEncoder,size(input,1));
    C1=C(end,:);
    %初始状态层
    for i1=1:length(args.decoderLayer)-2
        C2{i1}=C1*args.WeightTranR{i1}.w_k+args.WeightTranR{i1}.b_k;%zeros(1,size(args.WeightTranR{i1}.b_k,2));%
    end
    for i1=1:length(args.predictLayer)-2
        C3{i1}=C1*args.WeightTranP{i1}.w_k+args.WeightTranP{i1}.b_k;%zeros(1,size(args.WeightTranP{i1}.b_k,2));%
    end
    % decoder
    input_x11=[C1,zeros(1,size(input,2))];%t=1
    [x2,in2,f2,z2,c2,o2,y2,reconstruct]=LSTM_step_ff1(input_x11,C2,C1,args.WeightDecoder,size(input,1));
    % predict
    input_x11=[C1,zeros(1,size(label,2))];%t=1
    [x3,in3,f3,z3,c3,o3,y3,predict]=LSTM_step_ff1(input_x11,C3,C1,args.WeightPredict,size(label,1));
    
    %% 反向传播
    %% uncondictioned
    % predict and reconstruction layer
    [delta_up3,delta_c03,args.WeightPredict,args.Mom.WeightPredict]=...
        LSTM_step_bp1(args,label,predict,args.WeightPredict,args.Mom.WeightPredict,x3,in3,f3,z3,c3,o3,y3,C3);
    [delta_up2,delta_c02,args.WeightDecoder,args.Mom.WeightDecoder]=...
        LSTM_step_bp1(args,input(end:-1:1,:),reconstruct,args.WeightDecoder,args.Mom.WeightDecoder,x2,in2,f2,z2,c2,o2,y2,C2);
    %encoder layer
    temp1=zeros(1,size(C1,2));
    for i1=1:length(delta_c03) %计算C作为状态初始值时的梯度
        temp1=temp1+delta_c03{i1}*args.WeightTranP{i1}.w_k';
        dw_k=C1'*delta_c03{i1};
        args.Mom.WeightTranP{i1}.w_k=args.momentum*args.Mom.WeightTranP{i1}.w_k+dw_k;
        args.WeightTranP{i1}.w_k=args.WeightTranP{i1}.w_k-args.learningrate*args.Mom.WeightTranP{i1}.w_k;
        db_k=delta_c03{i1};
        args.Mom.WeightTranP{i1}.b_k=args.momentum*args.Mom.WeightTranP{i1}.b_k+db_k;
        args.WeightTranP{i1}.b_k=args.WeightTranP{i1}.b_k-args.learningrate*args.Mom.WeightTranP{i1}.b_k;
    end
    for i1=1:length(delta_c02)
        temp1=temp1+delta_c02{i1}*args.WeightTranR{i1}.w_k';
        dw_k=C1'*delta_c02{i1};
        args.Mom.WeightTranR{i1}.w_k=args.momentum*args.Mom.WeightTranR{i1}.w_k+dw_k;
        args.WeightTranR{i1}.w_k=args.WeightTranR{i1}.w_k-args.learningrate*args.Mom.WeightTranR{i1}.w_k;
        db_k=delta_c02{i1};
        args.Mom.WeightTranR{i1}.b_k=args.momentum*args.Mom.WeightTranR{i1}.b_k+db_k;
        args.WeightTranR{i1}.b_k=args.WeightTranR{i1}.b_k-args.learningrate*args.Mom.WeightTranR{i1}.b_k;
    end
    delta_k1=delta_up3+delta_up2+temp1;%
    dw_k1=y1{end}(end,:)'*delta_k1;
    db_k1=delta_k1;
    w_k1=args.WeightEncoder{end}.w_k;
    b_k1=args.WeightEncoder{end}.b_k;
    delta_up1=[zeros(size(input,1)-1,size(w_k1,1));delta_k1*w_k1'];
    args.Mom.WeightEncoder{end}.w_k=args.momentum*args.Mom.WeightEncoder{end}.w_k+dw_k1;
    args.Mom.WeightEncoder{end}.b_k=args.momentum*args.Mom.WeightEncoder{end}.b_k+db_k1;
    args.WeightEncoder{end}.w_k=w_k1-args.learningrate*args.Mom.WeightEncoder{end}.w_k;
    args.WeightEncoder{end}.b_k=b_k1-args.learningrate*args.Mom.WeightEncoder{end}.b_k;
    % lstm layer
    for i1=length(args.encoderLayer)-2:-1:1
        [delta_up1,WeightEncoder,MomWeightEncoder]=LSTM_step_bp(args,delta_up1,args.WeightEncoder{i1},args.Mom.WeightEncoder{i1},x1{i1},in1{i1},f1{i1},z1{i1},c1{i1},o1{i1},y1{i1});
        args.WeightEncoder{i1}=WeightEncoder;
        args.Mom.WeightEncoder{i1}=MomWeightEncoder;
    end

