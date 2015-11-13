function args=LSTM_train(args,input,label)
    for i1=1:args.maxecho
        for i2=1:length(input)
            [args]=LSTM_ff_bp(args,input{i2},label{i2});
        end
        %% Í³¼ÆÎó²î
        err=0;
        for i2=1:length(input)
            y=input{i2};
            for i1=1:length(args.layer)-2
                [~,~,~,~,~,~,y]=LSTM_ff(y,args,i1);
            end
            % softmax layer
            w_k=args.Weight{end}.w_k;
            temp=y*w_k;
            temp=exp(temp-max(temp,[],2)*ones(1,size(temp,2)));
            data_out =temp./(sum(temp,2)*ones(1,size(temp,2)));
            err=err-sum(sum(label{i2}.* log(data_out)));
        end
        fprintf('train error: %.4f \n',err);
    end


