function output=tanh_output_ff(w_k,b_k,x1)
[T]=size(x1,1);
output=tanh(x1*w_k+ones(T,1)*b_k);