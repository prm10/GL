function args=ae_gradient(args,layer_i,train_data)
w1=args.Weight{layer_i}.w;
w2=w1';
b1=args.Weight{layer_i}.b1;
b2=args.Weight{layer_i}.b2;
[numcases,numdims]=size(train_data);
%% Ç°Ïò´«²¥
a0 = train_data;
z1=a0*w1 + repmat(b1,numcases,1);
a1 = tanh(z1);
z2=a1*w2 + repmat(b2,numcases,1);
a2 = tanh(z2);

delta3=-(a0-a2).*dtanh(a2);
delta2=(delta3*w2').*dtanh(a1);

g0=(a1.*(1-a1)).^2;
g1=mean((1-2*a1).*g0).*sum(w1.^2);
g2=(ones(numdims,numcases)*g0)/numcases.*w1+(a0'*((1-2*a1).*g0))/numcases.*(ones(numdims,1)*sum(w1.^2));

db1=mean(delta2)+lambda*g1;
db2=mean(delta3);
dw1=a0'*delta2/numcases+lambda*g2;
dw2=a1'*delta3/numcases+lambda*w2;

args.Mom.Weight{layer_i}.w=args.momentum*args.Mom.Weight{layer_i}.w - args.learningrate*(dw1+dw2')/2;
args.Mom.Weight{layer_i}.b1 = args.momentum*args.Mom.Weight{layer_i}.b1 - args.learningrate*db1;
args.Mom.Weight{layer_i}.b2 = args.momentum*args.Mom.Weight{layer_i}.b2 - args.learningrate*db2;

args.Weight{layer_i}.w=args.Weight{layer_i}.w+args.Mom.Weight{layer_i}.w;
args.Weight{layer_i}.b1=args.Weight{layer_i}.b1+args.Mom.Weight{layer_i}.b1;
args.Weight{layer_i}.b2=args.Weight{layer_i}.b2+args.Mom.Weight{layer_i}.b2;

% function y=sigmoid(z)
%     y=1./(1+exp(-z));
% function y=dsigmoid(z)
%     y=z.*(1-z);
function y=dtanh(z)
    y=1-z.^2;