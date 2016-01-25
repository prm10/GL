function [adw]=ae_ff_bp(args,input,label)
%{
input: T*dim1
label: L*dim2
%}
    str_cmd = generate_cmd('1','args.WeightEncoder');
    eval(str_cmd);
    str_cmd = generate_cmd('2','args.WeightDecoder');
    eval(str_cmd);
    [T,~]=size(input);
    [L,dim2]=size(label);
    %% 前向传播
    %encoder
    x1=input;
    c0=zeros(1,size(r_i1,1));
    [x21,in21,f21,z21,c21,o21,y21]=LSTM_step_ff_fast(x1,c0,w_i1,r_i1,p_i1,w_f1,r_f1,p_f1,w_z1,r_z1,w_o1,r_o1,p_o1);
    C0=tanh_output_ff(w_k1,b_k1,y21(end,:));
    %transition
    cC0=tanh_output_ff(args.WeightStatic.w_c,args.WeightStatic.b_c,C0);
    %decoder
    x1=zeros(1,dim2);
    [x22,in22,f22,z22,c22,o22,y22,predict]=LSTM_step_ff_de(L,x1,cC0,C0,w_i2,r_i2,p_i2,w_f2,r_f2,p_f2,w_z2,r_z2,w_o2,r_o2,p_o2,w_k2,b_k2);
    %% 反向传播
    %decoder
    [delta_up,adw.WeightDecoder,delta_c0]=LSTM_step_bp_de(label,predict,w_i2,r_i2,p_i2,w_f2,r_f2,p_f2,w_z2,r_z2,w_o2,r_o2,p_o2,w_k2,b_k2,x22,in22,f22,z22,c22,o22,y22);
    %trainsition
    [delta_up_c0,dw_k,db_k]=tanh_output_bp(delta_c0,args.WeightStatic.w_c,C0,cC0);
    adw.WeightStatic.w_c=dw_k;
    adw.WeightStatic.b_c=db_k;

    delta_up=sum(delta_up(:,1:size(C0,2)));
    delta_up=delta_up+delta_up_c0;
    %encoder
    [delta_up,dw_k,db_k]=tanh_output_bp(delta_up,w_k1,y21(end,:),C0);
    delta_up=[zeros(T-1,size(dw_k,1));delta_up];
    [~,adw.WeightEncoder]=LSTM_step_bp_fast(delta_up,w_i1,r_i1,p_i1,w_f1,r_f1,p_f1,w_z1,r_z1,w_o1,r_o1,p_o1,x21,in21,f21,z21,c21,o21,y21);

    adw.WeightEncoder.w_k=dw_k;
    adw.WeightEncoder.b_k=db_k;

    adw.er=sum((label-predict).^2)/2/size(label,1)/size(label,2);
