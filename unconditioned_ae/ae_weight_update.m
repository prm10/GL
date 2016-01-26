function args=ae_weight_update(args,adw)
%% 权重衰减
tdw.WeightEncoder=weight_decay(args.WeightEncoder,args.Mom.WeightEncoder,args.momentum,args.weightDecay);
tdw.WeightStatic.w_c=args.momentum*args.Mom.WeightStatic.w_c+args.weightDecay*args.WeightStatic.w_c;
tdw.WeightStatic.b_c=args.momentum*args.Mom.WeightStatic.b_c;
tdw.WeightDecoder=weight_decay(args.WeightDecoder,args.Mom.WeightDecoder,args.momentum,args.weightDecay);
%% 限制梯度
% adw=clamp(adw,args.limit);
%%  加权求和
for i1=1:length(adw)
    tdw.WeightEncoder.w_i=tdw.WeightEncoder.w_i+(1-args.momentum)*adw{i1}.WeightEncoder.w_i/length(adw);
    tdw.WeightEncoder.r_i=tdw.WeightEncoder.r_i+(1-args.momentum)*adw{i1}.WeightEncoder.r_i/length(adw);
    tdw.WeightEncoder.p_i=tdw.WeightEncoder.p_i+(1-args.momentum)*adw{i1}.WeightEncoder.p_i/length(adw);
    tdw.WeightEncoder.w_f=tdw.WeightEncoder.w_f+(1-args.momentum)*adw{i1}.WeightEncoder.w_f/length(adw);
    tdw.WeightEncoder.r_f=tdw.WeightEncoder.r_f+(1-args.momentum)*adw{i1}.WeightEncoder.r_f/length(adw);
    tdw.WeightEncoder.p_f=tdw.WeightEncoder.p_f+(1-args.momentum)*adw{i1}.WeightEncoder.p_f/length(adw);
    tdw.WeightEncoder.w_z=tdw.WeightEncoder.w_z+(1-args.momentum)*adw{i1}.WeightEncoder.w_z/length(adw);
    tdw.WeightEncoder.r_z=tdw.WeightEncoder.r_z+(1-args.momentum)*adw{i1}.WeightEncoder.r_z/length(adw);
    tdw.WeightEncoder.w_o=tdw.WeightEncoder.w_o+(1-args.momentum)*adw{i1}.WeightEncoder.w_o/length(adw);
    tdw.WeightEncoder.r_o=tdw.WeightEncoder.r_o+(1-args.momentum)*adw{i1}.WeightEncoder.r_o/length(adw);
    tdw.WeightEncoder.p_o=tdw.WeightEncoder.p_o+(1-args.momentum)*adw{i1}.WeightEncoder.p_o/length(adw);
    tdw.WeightEncoder.w_k=tdw.WeightEncoder.w_k+(1-args.momentum)*adw{i1}.WeightEncoder.w_k/length(adw);
    tdw.WeightEncoder.b_k=tdw.WeightEncoder.b_k+(1-args.momentum)*adw{i1}.WeightEncoder.b_k/length(adw);

    tdw.WeightStatic.w_c=tdw.WeightStatic.w_c+(1-args.momentum)*adw{i1}.WeightStatic.w_c/length(adw);
    tdw.WeightStatic.b_c=tdw.WeightStatic.b_c+(1-args.momentum)*adw{i1}.WeightStatic.b_c/length(adw);

    tdw.WeightDecoder.w_i=tdw.WeightDecoder.w_i+(1-args.momentum)*adw{i1}.WeightDecoder.w_i/length(adw);
    tdw.WeightDecoder.r_i=tdw.WeightDecoder.r_i+(1-args.momentum)*adw{i1}.WeightDecoder.r_i/length(adw);
    tdw.WeightDecoder.p_i=tdw.WeightDecoder.p_i+(1-args.momentum)*adw{i1}.WeightDecoder.p_i/length(adw);
    tdw.WeightDecoder.w_f=tdw.WeightDecoder.w_f+(1-args.momentum)*adw{i1}.WeightDecoder.w_f/length(adw);
    tdw.WeightDecoder.r_f=tdw.WeightDecoder.r_f+(1-args.momentum)*adw{i1}.WeightDecoder.r_f/length(adw);
    tdw.WeightDecoder.p_f=tdw.WeightDecoder.p_f+(1-args.momentum)*adw{i1}.WeightDecoder.p_f/length(adw);
    tdw.WeightDecoder.w_z=tdw.WeightDecoder.w_z+(1-args.momentum)*adw{i1}.WeightDecoder.w_z/length(adw);
    tdw.WeightDecoder.r_z=tdw.WeightDecoder.r_z+(1-args.momentum)*adw{i1}.WeightDecoder.r_z/length(adw);
    tdw.WeightDecoder.w_o=tdw.WeightDecoder.w_o+(1-args.momentum)*adw{i1}.WeightDecoder.w_o/length(adw);
    tdw.WeightDecoder.r_o=tdw.WeightDecoder.r_o+(1-args.momentum)*adw{i1}.WeightDecoder.r_o/length(adw);
    tdw.WeightDecoder.p_o=tdw.WeightDecoder.p_o+(1-args.momentum)*adw{i1}.WeightDecoder.p_o/length(adw);
    tdw.WeightDecoder.w_k=tdw.WeightDecoder.w_k+(1-args.momentum)*adw{i1}.WeightDecoder.w_k/length(adw);
    tdw.WeightDecoder.b_k=tdw.WeightDecoder.b_k+(1-args.momentum)*adw{i1}.WeightDecoder.b_k/length(adw);
end
args.Mom=tdw;


%% 更新权值
args.WeightEncoder=update_weight(args.WeightEncoder,args.learningrate,tdw.WeightEncoder);
args.WeightDecoder=update_weight(args.WeightDecoder,args.learningrate,tdw.WeightDecoder);

args.WeightStatic.w_c=args.WeightStatic.w_c-args.learningrate*tdw.WeightStatic.w_c;
args.WeightStatic.b_c=args.WeightStatic.b_c-args.learningrate*tdw.WeightStatic.b_c;


function W=update_weight(W,lr,dW)
W.w_i=W.w_i-lr*dW.w_i;
W.r_i=W.r_i-lr*dW.r_i;
W.p_i=W.p_i-lr*dW.p_i;
W.w_f=W.w_f-lr*dW.w_f;
W.r_f=W.r_f-lr*dW.r_f;
W.p_f=W.p_f-lr*dW.p_f;
W.w_z=W.w_z-lr*dW.w_z;
W.r_z=W.r_z-lr*dW.r_z;
W.w_o=W.w_o-lr*dW.w_o;
W.r_o=W.r_o-lr*dW.r_o;
W.p_o=W.p_o-lr*dW.p_o;
W.w_k=W.w_k-lr*dW.w_k;
W.b_k=W.b_k-lr*dW.b_k;

function MW=weight_decay(W,MW,momentum,wd)
MW.w_i=momentum*MW.w_i+wd*W.w_i;
MW.r_i=momentum*MW.r_i+wd*W.r_i;
MW.p_i=momentum*MW.p_i+wd*W.p_i;
MW.w_f=momentum*MW.w_f+wd*W.w_f;
MW.r_f=momentum*MW.r_f+wd*W.r_f;
MW.p_f=momentum*MW.p_f+wd*W.p_f;
MW.w_z=momentum*MW.w_z+wd*W.w_z;
MW.r_z=momentum*MW.r_z+wd*W.r_z;
MW.w_o=momentum*MW.w_o+wd*W.w_o;
MW.r_o=momentum*MW.r_o+wd*W.r_o;
MW.p_o=momentum*MW.p_o+wd*W.p_o;
MW.w_k=momentum*MW.w_k+wd*W.w_k;
MW.b_k=momentum*MW.b_k;
