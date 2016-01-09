function [predict,error]=fsc_ff(input0,label0,args)
error=0;
for i2=1:length(input0)
    input=input0{i2};
    label=label0{i2};
    %% 前向传播
    % encoder
   %
    input_x11=input(1,:);%t=1
    [~,~,~,~,~,~,~,predict]=LSTM_step_ff1(input_x11,0,input,args.Weight,size(input,1),args.outputLayer);
%{
    x1=input;
    c0=zeros(1,size(args.Weight{1}.r_i,1));
    outputLayer=args.outputLayer;
    
    w_i=args.Weight{1}.w_i;
	r_i=args.Weight{1}.r_i;
	p_i=args.Weight{1}.p_i;
	w_f=args.Weight{1}.w_f;
	r_f=args.Weight{1}.r_f;
	p_f=args.Weight{1}.p_f;
	w_z=args.Weight{1}.w_z;
	r_z=args.Weight{1}.r_z;
	w_o=args.Weight{1}.w_o;
	r_o=args.Weight{1}.r_o;
	p_o=args.Weight{1}.p_o;
    w_k=args.Weight{end}.w_k;
    b_k=args.Weight{end}.b_k;

    [~,~,~,~,~,~,~,predict]=LSTM_step_ff_fast(x1,c0,outputLayer,...
    w_i,r_i,p_i,...
    w_f,r_f,p_f,...
    w_z,r_z,...
    w_o,r_o,p_o,...
    w_k,b_k);
%}
    % 计算误差
    switch args.outputLayer
        case{'softmax'}
            error=error-sum(sum(label.*log(predict)))/size(predict,1)*2;
        otherwise
            error=error+sum(sum((label-predict).^2))/size(predict,1)/size(predict,2);
    end
end
error=error/length(input0)/2;