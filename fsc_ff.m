function [predict,error]=fsc_ff(input0,label0,args)
error=0;
for i2=1:length(input0)
    input=input0{i2};
    label=label0{i2};
    %% ǰ�򴫲�
    % encoder
%{
    input_x11=input(1,:);%t=1
    [~,~,~,~,~,~,~,predict]=LSTM_step_ff1(input_x11,0,input,args.Weight,size(input,1),args.outputLayer);
%}
    %{
    x1=input;
    c0=zeros(1,size(args.Weight{1}.r_i,1));
    [~,~,~,~,~,~,y2]=LSTM_step_ff_fast(x1,c0,args.Weight{1});
    predict=LSTM_output_ff(args.outputLayer,args.Weight{end}.w_k,args.Weight{end}.b_k,y2);
%}
    x1{1}=input;
    for i1=1:length(args.layer)-2
        c0=zeros(1,size(args.Weight{i1}.r_i,1));
        [~,~,~,~,~,~,y2]=LSTM_step_ff_fast(x1{i1},c0,args.Weight{i1});
        predict=LSTM_output_ff(args.outputLayer,args.Weight{end}.w_k,args.Weight{end}.b_k,y2);
    end
    % �������
    switch args.outputLayer
        case{'softmax'}
            error=error-sum(sum(label.*log(predict)))/size(predict,1)*2;
        otherwise
            error=error+sum(sum((label-predict).^2))/size(predict,1)/size(predict,2);
    end
end
error=error/length(input0)/2;