function [a2,error]=ae_ff(test_data,args,layer_i)
w1=args.Weight{layer_i}.w;
w2=w1';
b1=args.Weight{layer_i}.b1;
b2=args.Weight{layer_i}.b2;
[numcases,numdims]=size(test_data);

a0 = test_data;
z1=a0*w1 + repmat(b1,numcases,1);
a1 = tanh(z1);
z2=a1*w2 + repmat(b2,numcases,1);
a2 = tanh(z2);

%% ÊÕ¼¯Îó²î
error=sum(sum((a0-a2).^2))/numdims/numcases/2;