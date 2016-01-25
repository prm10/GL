function [predict,error]=ae_ff(input0,label0,args)
error=0;
for i2=1:length(input0)
    input=input0{i2};
    label=label0{i2};
    
    str_cmd = generate_cmd('1','args.WeightEncoder');
    eval(str_cmd);
    str_cmd = generate_cmd('2','args.WeightDecoder');
    eval(str_cmd);
    [T,dim1]=size(input);
    [L,dim2]=size(label);
    %% 前向传播
    %encoder
    x1=input;
    c0=zeros(1,size(r_i1,1));
    [~,~,~,~,~,~,y21]=LSTM_step_ff_fast(x1,c0,w_i1,r_i1,p_i1,w_f1,r_f1,p_f1,w_z1,r_z1,w_o1,r_o1,p_o1);
    C0=tanh_output_ff(w_k1,b_k1,y21(end,:));
    %transition
    cC0=tanh_output_ff(args.WeightStatic.w_c,args.WeightStatic.b_c,C0);
    %decoder
    x1=zeros(1,dim2);
    [~,~,~,~,~,~,~,predict]=LSTM_step_ff_de(L,x1,cC0,C0,w_i2,r_i2,p_i2,w_f2,r_f2,p_f2,w_z2,r_z2,w_o2,r_o2,p_o2,w_k2,b_k2);

    % 计算误差
    switch args.outputLayer
        case{'softmax'}
            error=error-sum(sum(label.*log(predict)))/size(predict,1);
        otherwise
            error=error+sum(sum((label-predict).^2))/size(predict,1)/size(predict,2)/2;
    end
end
error=error/length(input0);