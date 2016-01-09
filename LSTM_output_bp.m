function [delta_down,dw_k,db_k]=LSTM_output_bp(outputLayer,w_k,y2_end,label,predict)
switch outputLayer
    case{'softmax'}
        delta_k=-(label-predict)/size(predict,1);
    otherwise
        delta_k=-(label-predict)/size(predict,1).*(1-predict.^2);
end
dw_k=y2_end'*delta_k;
db_k=sum(delta_k);
delta_down=delta_k*w_k';
