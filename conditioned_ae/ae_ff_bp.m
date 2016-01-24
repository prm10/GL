function [adw]=ae_ff_bp(args,input,label)
%{
input: T*dim1
label: L*dim2
%}
[T,dim1]=size(input);
[L,dim2]=size(label);
    %% 前向传播
    %encoder
    x1=input;
    for i1=1:length(args.layerEncoder)-2
        c0=zeros(1,size(args.WeightEncoder{i1}.r_i,1));
        [x21{i1},in21{i1},f21{i1},z21{i1},c21{i1},o21{i1},y21{i1}]=LSTM_step_ff_fast(x1,c0,args.WeightEncoder{i1});
        x1=y21{i1};
    end
    cEncoder=tanh_output_ff(args.WeightEncoder{end}.w_k,args.WeightEncoder{end}.b_k,y21{end}(end,:));
    %transition
    cStatic=tanh_output_ff(args.WeightStatic.w_k1,args.WeightStatic.b_k1,cEncoder);
    cDecoder=tanh_output_ff(args.WeightStatic.w_k2,args.WeightStatic.b_k2,cStatic);
    for i1=1:length(args.WeightStatic.b_c)
        cC0{i1}=tanh_output_ff(args.WeightStatic.w_c{i1},args.WeightStatic.b_c{i1},cStatic);
    end
    %decoder
    x1=[ones(L,1)*cDecoder,[zeros(1,dim1);label(1:end-1,:)]];
    for i1=1:length(args.layerDecoder)-2
%         c0=zeros(1,size(args.WeightDecoder{i1}.r_i,1));
        [x22{i1},in22{i1},f22{i1},z22{i1},c22{i1},o22{i1},y22{i1}]=LSTM_step_ff_fast(x1,cC0{i1},args.WeightDecoder{i1});
        x1=y22{i1};
    end
    predict=tanh_output_ff(args.WeightDecoder{end}.w_k,args.WeightDecoder{end}.b_k,y22{end});
    %% 反向传播
    %decoder
    delta_up=-(label-predict)/size(predict,1);
    [delta_up,dw_k,db_k]=tanh_output_bp(delta_up,args.WeightDecoder{end}.w_k,y22{end},predict);
    adw.WeightDecoder{length(args.layerDecoder)-1}.w_k=dw_k;
    adw.WeightDecoder{length(args.layerDecoder)-1}.b_k=db_k;
    for i1=length(args.layerDecoder)-2:-1:1
        [delta_up,adw.WeightDecoder{i1},delta_c0{i1}]=LSTM_step_bp_fast(delta_up,args.WeightDecoder{i1},x22{i1},in22{i1},f22{i1},z22{i1},c22{i1},o22{i1},y22{i1});
    end
    %trainsition
    delta_up=sum(delta_up(:,1:size(cDecoder,2)));
    [delta_up,dw_k,db_k]=tanh_output_bp(delta_up,args.WeightStatic.w_k2,cStatic,cDecoder);
    adw.WeightStatic.w_k2=dw_k;
    adw.WeightStatic.b_k2=db_k;
    [delta_up,dw_k,db_k]=tanh_output_bp(delta_up,args.WeightStatic.w_k1,cEncoder,cStatic);
    adw.WeightStatic.w_k1=dw_k;
    adw.WeightStatic.b_k1=db_k;
    for i1=1:
    %encoder
%     [delta_up,dw_k,db_k]=LSTM_output_bp(args.outputLayer,args.WeightEncoder{end}.w_k,y21{end}(end,:),delta_up+cEncoder,cEncoder);
    [delta_up,dw_k,db_k]=tanh_output_bp(delta_up,args.WeightEncoder{end}.w_k,y21{end}(end,:),cEncoder);
    adw.WeightEncoder{length(args.layerEncoder)-1}.w_k=dw_k;
    adw.WeightEncoder{length(args.layerEncoder)-1}.b_k=db_k;
    delta_up=[zeros(T-1,size(dw_k,1));delta_up];
    for i1=length(args.layerEncoder)-2:-1:1
        [delta_up,adw.WeightEncoder{i1}]=LSTM_step_bp_fast(delta_up,args.WeightEncoder{i1},x21{i1},in21{i1},f21{i1},z21{i1},c21{i1},o21{i1},y21{i1});
    end
    
    
