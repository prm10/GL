function [delta_down,dw_k,db_k]=tanh_output_bp(delta_up,w_k,x1,y1)
delta_k=delta_up.*(1-y1.^2);
dw_k=x1'*delta_k;
db_k=sum(delta_k,1);
delta_down=delta_k*w_k';