function [data_out,error]=LSTM_ff(input,label,args)
error=0;
for i2=1:length(input)
    y=input{i2};
    for layer_i=1:length(args.layer)-2
        [~,~,~,~,~,~,y]=LSTM_step_ff(y,args,layer_i);
    end
    w_k=args.Weight{end}.w_k;
    z_k=y*w_k;
    switch args.outputtype
        case 'softmax'
            % softmax layer
            temp=exp(z_k-max(z_k,[],2)*ones(1,size(z_k,2)));
            data_out =temp./(sum(temp,2)*ones(1,size(temp,2)));
            error=error-sum(sum(label{i2}.* log(data_out)))/size(data_out,1);
        case 'linear'
            data_out=z_k;
            error=error+sum(sum((label{i2}-data_out).^2))/size(data_out,1)/size(data_out,2);
    end
end
error=error/length(input)*100;