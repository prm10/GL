function [predict,error]=fsc_ff(input0,label0,args)
error=0;
for i2=1:length(input0)
    input=input0{i2};
    label=label0{i2};
    [T,dim1]=size(input);
    [L,dim2]=size(label);
    %% 前向传播
    %encoder
    x1=input;
    for i1=1:length(args.layerEncoder)-2
        c0=zeros(1,size(args.WeightEncoder{i1}.r_i,1));
        [~,~,~,~,~,~,y2]=LSTM_step_ff_fast(x1,c0,args.WeightEncoder{i1});
        x1=y2;
    end
    cEncoder=tanh_output_ff(args.WeightEncoder{end}.w_k,args.WeightEncoder{end}.b_k,y2(end,:));
    %transition
    cDecoder=tanh_output_ff(args.WeightStatic.w_k,args.WeightStatic.b_k,cEncoder);
    cC0=cell(length(args.WeightStatic.b_c));
    for i1=1:length(args.WeightStatic.b_c)
        cC0{i1}=tanh_output_ff(args.WeightStatic.w_c{i1},args.WeightStatic.b_c{i1},cEncoder);
    end
    %decoder
    x1=[ones(L,1)*cDecoder,[zeros(1,dim1);label(1:end-1,:)]];
    for i1=1:length(args.layerDecoder)-2
%         c0=zeros(1,size(args.WeightDecoder{i1}.r_i,1));
        [~,~,~,~,~,~,y2]=LSTM_step_ff_fast(x1,cC0{i1},args.WeightDecoder{i1});
        x1=y2;
    end
    predict=tanh_output_ff(args.WeightDecoder{end}.w_k,args.WeightDecoder{end}.b_k,y2);
    
    % 计算误差
    switch args.outputLayer
        case{'softmax'}
            error=error-sum(sum(label.*log(predict)))/size(predict,1);
        otherwise
            error=error+sum(sum((label-predict).^2))/size(predict,1)/size(predict,2)/2;
    end
end
error=error/length(input0);