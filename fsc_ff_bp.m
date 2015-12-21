function [adw]=fsc_ff_bp(args,input,label)
    %% 前向传播
    % encoder
    input_x11=input(1,:);%t=1
    [x1,in1,f1,z1,c1,o1,y1,C]=LSTM_step_ff1(input_x11,0,input,args.Weight,size(input,1),args.outputLayer);
    %% 反向传播
    switch args.outputLayer
        case{'softmax'}
            delta_k=-(label-C)/size(C,1);
        otherwise
            delta_k=-(label-C)/size(C,1).*(1-C.^2);
    end
    dw_k1=y1{end}'*delta_k;
    db_k1=sum(delta_k);
    w_k1=args.Weight{end}.w_k;
    delta_up=delta_k*w_k1';
    adw.Weight{length(args.layer)-1}.w_k=dw_k1;
    adw.Weight{length(args.layer)-1}.b_k=db_k1;
    % lstm layer
    for i1=length(args.layer)-2:-1:1
        [delta_up,adw.Weight{i1}]=LSTM_step_bp(args,delta_up,args.Weight{i1},args.Mom.Weight{i1},x1{i1},in1{i1},f1{i1},z1{i1},c1{i1},o1{i1},y1{i1});
    end

