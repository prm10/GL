function [args]=LSTM_ff_bp(args,input,label)
    %%前向传播
    x0=input;
    for i1=1:length(args.layer)-2
        [x{i1},in2{i1},f2{i1},z2{i1},c{i1},o2{i1},y{i1}]=LSTM_step_ff(x0,args,i1);
        x0=y{i1};
    end
    %最后一层
    w_k=args.Weight{end}.w_k;
    z_k=y{end}*w_k;
    switch args.outputtype
        case 'softmax'
            % softmax layer
            temp=exp(z_k-max(z_k,[],2)*ones(1,size(z_k,2)));
            data_out =temp./(sum(temp,2)*ones(1,size(temp,2)));
%             err=-sum(sum(label.* log(data_out)));
        case 'tanh'
            data_out=tanh(z_k);
        case 'linear'
            data_out=z_k;
    end
    
    %% 反向传播
    switch args.outputtype
        case 'softmax'
            delta_k=-(label-data_out);
        case 'linear'
            delta_k=-(label-data_out);
        case 'tanh'
            delta_k=-(label-data_out).*(1-data_out.^2);
    end

    dw_k=y{end}'*delta_k;
    if(exist('args.D.w_k','var'))
        args.D.w_k=args.momentum*args.D.w_k+dw_k;
    else
        args.D.w_k=dw_k;
    end
    % LSTM layer
    delta_up=delta_k*w_k';
    for i1=length(args.layer)-2:-1:1
        [delta_up,args]=LSTM_step_bp(delta_up,args,i1,x{i1},in2{i1},f2{i1},z2{i1},c{i1},o2{i1},y{i1});
    end
    % learning rate
    learningrate=args.learningrate;
    args.Weight{end}.w_k=w_k-learningrate*args.D.w_k;
