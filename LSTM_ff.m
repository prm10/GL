function [data_out,error]=LSTM_ff(input,label,args)
error=0;
for i2=1:length(input)
    y=input{i2};
    for layer_i=1:length(args.layer)-2
        [~,~,~,~,~,~,y]=LSTM_step_ff(y,args,layer_i);
    end
    % softmax layer
    w_k=args.Weight{end}.w_k;
    temp=y*w_k;
    temp=exp(temp-max(temp,[],2)*ones(1,size(temp,2)));
    data_out =temp./(sum(temp,2)*ones(1,size(temp,2)));
    error=error-sum(sum(label{i2}.* log(data_out)))/size(data_out,1);
end
error=error/length(input)*100;