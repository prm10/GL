function args=ae_weight_update(args,adw)
%% 权重衰减
tdw.WeightEncoder=weight_decay(args.WeightEncoder,args.Mom.WeightEncoder,args.momentum,args.weightDecay);
tdw.WeightStatic.w_k1=args.momentum*args.Mom.WeightStatic.w_k1+args.weightDecay*args.WeightStatic.w_k1;
tdw.WeightStatic.b_k1=args.momentum*args.Mom.WeightStatic.b_k1;
tdw.WeightStatic.w_k2=args.momentum*args.Mom.WeightStatic.w_k2+args.weightDecay*args.WeightStatic.w_k2;
tdw.WeightStatic.b_k2=args.momentum*args.Mom.WeightStatic.b_k2;
tdw.WeightDecoder=weight_decay(args.WeightDecoder,args.Mom.WeightDecoder,args.momentum,args.weightDecay);
%% 限制梯度
% adw=clamp(adw,args.limit);
%%  加权求和
for i1=1:length(adw)
    for i2=1:length(tdw.WeightEncoder)-1
        tdw.WeightEncoder{i2}.w_i=tdw.WeightEncoder{i2}.w_i+adw{i1}.WeightEncoder{i2}.w_i/length(adw);
        tdw.WeightEncoder{i2}.r_i=tdw.WeightEncoder{i2}.r_i+adw{i1}.WeightEncoder{i2}.r_i/length(adw);
        tdw.WeightEncoder{i2}.p_i=tdw.WeightEncoder{i2}.p_i+adw{i1}.WeightEncoder{i2}.p_i/length(adw);
        tdw.WeightEncoder{i2}.w_f=tdw.WeightEncoder{i2}.w_f+adw{i1}.WeightEncoder{i2}.w_f/length(adw);
        tdw.WeightEncoder{i2}.r_f=tdw.WeightEncoder{i2}.r_f+adw{i1}.WeightEncoder{i2}.r_f/length(adw);
        tdw.WeightEncoder{i2}.p_f=tdw.WeightEncoder{i2}.p_f+adw{i1}.WeightEncoder{i2}.p_f/length(adw);
        tdw.WeightEncoder{i2}.w_z=tdw.WeightEncoder{i2}.w_z+adw{i1}.WeightEncoder{i2}.w_z/length(adw);
        tdw.WeightEncoder{i2}.r_z=tdw.WeightEncoder{i2}.r_z+adw{i1}.WeightEncoder{i2}.r_z/length(adw);
        tdw.WeightEncoder{i2}.w_o=tdw.WeightEncoder{i2}.w_o+adw{i1}.WeightEncoder{i2}.w_o/length(adw);
        tdw.WeightEncoder{i2}.r_o=tdw.WeightEncoder{i2}.r_o+adw{i1}.WeightEncoder{i2}.r_o/length(adw);
        tdw.WeightEncoder{i2}.p_o=tdw.WeightEncoder{i2}.p_o+adw{i1}.WeightEncoder{i2}.p_o/length(adw);
    end
    tdw.WeightEncoder{end}.w_k=tdw.WeightEncoder{end}.w_k+adw{i1}.WeightEncoder{end}.w_k/length(adw);
    tdw.WeightEncoder{end}.b_k=tdw.WeightEncoder{end}.b_k+adw{i1}.WeightEncoder{end}.b_k/length(adw);
	
    tdw.WeightStatic.w_k1=tdw.WeightStatic.w_k1+adw{i1}.WeightStatic.w_k1/length(adw);
    tdw.WeightStatic.b_k1=tdw.WeightStatic.b_k1+adw{i1}.WeightStatic.b_k1/length(adw);
	tdw.WeightStatic.w_k2=tdw.WeightStatic.w_k2+adw{i1}.WeightStatic.w_k2/length(adw);
    tdw.WeightStatic.b_k2=tdw.WeightStatic.b_k2+adw{i1}.WeightStatic.b_k2/length(adw);
	
    for i2=1:length(tdw.WeightDecoder)-1
        tdw.WeightDecoder{i2}.w_i=tdw.WeightDecoder{i2}.w_i+adw{i1}.WeightDecoder{i2}.w_i/length(adw);
        tdw.WeightDecoder{i2}.r_i=tdw.WeightDecoder{i2}.r_i+adw{i1}.WeightDecoder{i2}.r_i/length(adw);
        tdw.WeightDecoder{i2}.p_i=tdw.WeightDecoder{i2}.p_i+adw{i1}.WeightDecoder{i2}.p_i/length(adw);
        tdw.WeightDecoder{i2}.w_f=tdw.WeightDecoder{i2}.w_f+adw{i1}.WeightDecoder{i2}.w_f/length(adw);
        tdw.WeightDecoder{i2}.r_f=tdw.WeightDecoder{i2}.r_f+adw{i1}.WeightDecoder{i2}.r_f/length(adw);
        tdw.WeightDecoder{i2}.p_f=tdw.WeightDecoder{i2}.p_f+adw{i1}.WeightDecoder{i2}.p_f/length(adw);
        tdw.WeightDecoder{i2}.w_z=tdw.WeightDecoder{i2}.w_z+adw{i1}.WeightDecoder{i2}.w_z/length(adw);
        tdw.WeightDecoder{i2}.r_z=tdw.WeightDecoder{i2}.r_z+adw{i1}.WeightDecoder{i2}.r_z/length(adw);
        tdw.WeightDecoder{i2}.w_o=tdw.WeightDecoder{i2}.w_o+adw{i1}.WeightDecoder{i2}.w_o/length(adw);
        tdw.WeightDecoder{i2}.r_o=tdw.WeightDecoder{i2}.r_o+adw{i1}.WeightDecoder{i2}.r_o/length(adw);
        tdw.WeightDecoder{i2}.p_o=tdw.WeightDecoder{i2}.p_o+adw{i1}.WeightDecoder{i2}.p_o/length(adw);
    end
    tdw.WeightDecoder{end}.w_k=tdw.WeightDecoder{end}.w_k+adw{i1}.WeightDecoder{end}.w_k/length(adw);
    tdw.WeightDecoder{end}.b_k=tdw.WeightDecoder{end}.b_k+adw{i1}.WeightDecoder{end}.b_k/length(adw);
end
args.Mom=tdw;


%% 更新权值
args.WeightEncoder=update_weight(args.WeightEncoder,args.learningrate,tdw.WeightEncoder);
args.WeightDecoder=update_weight(args.WeightDecoder,args.learningrate,tdw.WeightDecoder);
args.WeightStatic.w_k1=args.WeightStatic.w_k1-args.learningrate*tdw.WeightStatic.w_k1;
args.WeightStatic.b_k1=args.WeightStatic.b_k1-args.learningrate*tdw.WeightStatic.b_k1;
args.WeightStatic.w_k2=args.WeightStatic.w_k2-args.learningrate*tdw.WeightStatic.w_k2;
args.WeightStatic.b_k2=args.WeightStatic.b_k2-args.learningrate*tdw.WeightStatic.b_k2;

function W=update_weight(W,lr,dW)
for i2=1:length(dW)-1
    W{i2}.w_i=W{i2}.w_i-lr*dW{i2}.w_i;
    W{i2}.r_i=W{i2}.r_i-lr*dW{i2}.r_i;
    W{i2}.p_i=W{i2}.p_i-lr*dW{i2}.p_i;
    W{i2}.w_f=W{i2}.w_f-lr*dW{i2}.w_f;
    W{i2}.r_f=W{i2}.r_f-lr*dW{i2}.r_f;
    W{i2}.p_f=W{i2}.p_f-lr*dW{i2}.p_f;
    W{i2}.w_z=W{i2}.w_z-lr*dW{i2}.w_z;
    W{i2}.r_z=W{i2}.r_z-lr*dW{i2}.r_z;
    W{i2}.w_o=W{i2}.w_o-lr*dW{i2}.w_o;
    W{i2}.r_o=W{i2}.r_o-lr*dW{i2}.r_o;
    W{i2}.p_o=W{i2}.p_o-lr*dW{i2}.p_o;
end
W{end}.w_k=W{end}.w_k-lr*dW{end}.w_k;
W{end}.b_k=W{end}.b_k-lr*dW{end}.b_k;

function MW=weight_decay(W,MW,momentum,wd)
% 动量衰减
for i2=1:length(MW)-1
	MW{i2}.w_i=momentum*MW{i2}.w_i+wd*W{i2}.w_i;
	MW{i2}.r_i=momentum*MW{i2}.r_i+wd*W{i2}.r_i;
	MW{i2}.p_i=momentum*MW{i2}.p_i+wd*W{i2}.p_i;
	MW{i2}.w_f=momentum*MW{i2}.w_f+wd*W{i2}.w_f;
	MW{i2}.r_f=momentum*MW{i2}.r_f+wd*W{i2}.r_f;
	MW{i2}.p_f=momentum*MW{i2}.p_f+wd*W{i2}.p_f;
	MW{i2}.w_z=momentum*MW{i2}.w_z+wd*W{i2}.w_z;
	MW{i2}.r_z=momentum*MW{i2}.r_z+wd*W{i2}.r_z;
	MW{i2}.w_o=momentum*MW{i2}.w_o+wd*W{i2}.w_o;
	MW{i2}.r_o=momentum*MW{i2}.r_o+wd*W{i2}.r_o;
	MW{i2}.p_o=momentum*MW{i2}.p_o+wd*W{i2}.p_o;
end
MW{end}.w_k=momentum*MW{end}.w_k+wd*W{end}.w_k;
MW{end}.b_k=momentum*MW{end}.b_k;