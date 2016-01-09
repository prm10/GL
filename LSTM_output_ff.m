function output=LSTM_output_ff(outputLayer,w_k,b_k,y2)
% outputLayer=args.outputLayer;
% w_k=W{end}.w_k;
% b_k=W{end}.b_k;
%×îºóÒ»²ã
T=size(y2,1);
z_k2=y2*w_k+ones(T,1)*b_k;
numClass=size(w_k,2);
switch outputLayer
    case{'softmax'}
        temp=exp(z_k2-max(z_k2,[],2)*ones(1,numClass));
        output=temp./(sum(temp,2)*ones(1,numClass));
    otherwise
        output=tanh(z_k2);
end