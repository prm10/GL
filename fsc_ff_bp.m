function [adw]=fsc_ff_bp(args,input,label)
    %% 前向传播
    x1=input;
    for i1=1:length(args.layer)-2
        c0=zeros(1,size(args.Weight{i1}.r_i,1));
        [x2{i1},in2{i1},f2{i1},z2{i1},c2{i1},o2{i1},y2{i1}]=LSTM_step_ff_fast(x1,c0,args.Weight{i1});
        x1=y2{i1};
    end
    predict=LSTM_output_ff(args.outputLayer,args.Weight{end}.w_k,args.Weight{end}.b_k,y2{end});
    %% 反向传播
     [delta_up,dw_k,db_k]=LSTM_output_bp(args.outputLayer,args.Weight{end}.w_k,y2{end},label,predict);
    adw.Weight{length(args.layer)-1}.w_k=dw_k;
    adw.Weight{length(args.layer)-1}.b_k=db_k;
    % lstm layer
    for i1=length(args.layer)-2:-1:1
        [delta_up,adw.Weight{i1}]=LSTM_step_bp_fast(delta_up,args.Weight{i1},x2{i1},in2{i1},f2{i1},z2{i1},c2{i1},o2{i1},y2{i1});
    end
