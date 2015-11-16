function [args]=LSTM_ff_bp(args,input,label)
    % data
    x0=input;
    for i1=1:length(args.layer)-2
        [x{i1},in2{i1},f2{i1},z2{i1},c{i1},o2{i1},y{i1}]=LSTM_step_ff(x0,args,i1);
        x0=y{i1};
    end
    % softmax layer
    w_k=args.Weight{end}.w_k;
    temp=y{end}*w_k;
    temp=exp(temp-max(temp,[],2)*ones(1,size(temp,2)));
    data_out =temp./(sum(temp,2)*ones(1,size(temp,2)));
    err=-sum(sum(label.* log(data_out)));
    %% ·´Ïò´«²¥
    % softmax layer
    delta_k=-(label-data_out);
    dw_k=y{end}'*delta_k;

    % LSTM layer
    delta_up=delta_k*w_k';
    for i1=length(args.layer)-2:-1:1
        [delta_up,args]=LSTM_step_bp(delta_up,args,i1,x{i1},in2{i1},f2{i1},z2{i1},c{i1},o2{i1},y{i1});
    end
    % learning rate
    learningrate=args.learningrate;
    args.Weight{end}.w_k=w_k-learningrate*dw_k;
