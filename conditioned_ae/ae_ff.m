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
    cEncoder=LSTM_output_ff(args.outputLayer,args.WeightEncoder{end}.w_k,args.WeightEncoder{end}.b_k,y2(end,:));
    %transition
    cStatic=LSTM_output_ff(args.outputLayer,args.WeightStatic.w_k1,args.WeightStatic.b_k1,cEncoder);
    cDecoder=LSTM_output_ff(args.outputLayer,args.WeightStatic.w_k2,args.WeightStatic.b_k2,cStatic);
    %decoder
    x1=[ones(L,1)*cDecoder,[zeros(1,dim1);label(1:end-1,:)]];
    for i1=1:length(args.layerDecoder)-2
        c0=zeros(1,size(args.WeightDecoder{i1}.r_i,1));
        [~,~,~,~,~,~,y2]=LSTM_step_ff_fast(x1,c0,args.WeightDecoder{i1});
        x1=y2;
    end
    predict=LSTM_output_ff(args.outputLayer,args.WeightDecoder{end}.w_k,args.WeightDecoder{end}.b_k,y2);
    
    % 计算误差
    switch args.outputLayer
        case{'softmax'}
            error=error-sum(sum(label.*log(predict)))/size(predict,1)*2;
        otherwise
            error=error+sum(sum((label-predict).^2))/size(predict,1)/size(predict,2);
    end
end
error=error/length(input0)/2;