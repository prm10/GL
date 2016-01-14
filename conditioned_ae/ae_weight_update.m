function args=ae_weight_update(args,adw)
tdw.Weight=weight_decay(args.Weight,args.Mom.Weight,args.momentum,args.weightDecay);

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
MW{end}.b_k=momentum*MW{end}.b_k+wd*W{end}.b_k;


function tdw=add_weight(tdw,adw)
%%  加权求和
for i1=1:length(adw)
    for i2=1:length(tdw.Weight)-1
        tdw.Weight{i2}.w_i=tdw.Weight{i2}.w_i+adw{i1}.Weight{i2}.w_i/length(adw);
        tdw.Weight{i2}.r_i=tdw.Weight{i2}.r_i+adw{i1}.Weight{i2}.r_i/length(adw);
        tdw.Weight{i2}.p_i=tdw.Weight{i2}.p_i+adw{i1}.Weight{i2}.p_i/length(adw);
        tdw.Weight{i2}.w_f=tdw.Weight{i2}.w_f+adw{i1}.Weight{i2}.w_f/length(adw);
        tdw.Weight{i2}.r_f=tdw.Weight{i2}.r_f+adw{i1}.Weight{i2}.r_f/length(adw);
        tdw.Weight{i2}.p_f=tdw.Weight{i2}.p_f+adw{i1}.Weight{i2}.p_f/length(adw);
        tdw.Weight{i2}.w_z=tdw.Weight{i2}.w_z+adw{i1}.Weight{i2}.w_z/length(adw);
        tdw.Weight{i2}.r_z=tdw.Weight{i2}.r_z+adw{i1}.Weight{i2}.r_z/length(adw);
        tdw.Weight{i2}.w_o=tdw.Weight{i2}.w_o+adw{i1}.Weight{i2}.w_o/length(adw);
        tdw.Weight{i2}.r_o=tdw.Weight{i2}.r_o+adw{i1}.Weight{i2}.r_o/length(adw);
        tdw.Weight{i2}.p_o=tdw.Weight{i2}.p_o+adw{i1}.Weight{i2}.p_o/length(adw);
    end
    tdw.Weight{end}.w_k=tdw.Weight{end}.w_k+adw{i1}.Weight{end}.w_k/length(adw);
    tdw.Weight{end}.b_k=tdw.Weight{end}.b_k+adw{i1}.Weight{end}.b_k/length(adw);
end
args.Mom=tdw;

function W=update_weight(W,lr,tdw)
%% 更新权值
lr=args.learningrate;
for i2=1:length(tdw.Weight)-1
    args.Weight{i2}.w_i=args.Weight{i2}.w_i-lr*tdw.Weight{i2}.w_i;
    args.Weight{i2}.r_i=args.Weight{i2}.r_i-lr*tdw.Weight{i2}.r_i;
    args.Weight{i2}.p_i=args.Weight{i2}.p_i-lr*tdw.Weight{i2}.p_i;
    args.Weight{i2}.w_f=args.Weight{i2}.w_f-lr*tdw.Weight{i2}.w_f;
    args.Weight{i2}.r_f=args.Weight{i2}.r_f-lr*tdw.Weight{i2}.r_f;
    args.Weight{i2}.p_f=args.Weight{i2}.p_f-lr*tdw.Weight{i2}.p_f;
    args.Weight{i2}.w_z=args.Weight{i2}.w_z-lr*tdw.Weight{i2}.w_z;
    args.Weight{i2}.r_z=args.Weight{i2}.r_z-lr*tdw.Weight{i2}.r_z;
    args.Weight{i2}.w_o=args.Weight{i2}.w_o-lr*tdw.Weight{i2}.w_o;
    args.Weight{i2}.r_o=args.Weight{i2}.r_o-lr*tdw.Weight{i2}.r_o;
    args.Weight{i2}.p_o=args.Weight{i2}.p_o-lr*tdw.Weight{i2}.p_o;
end
args.Weight{end}.w_k=args.Weight{end}.w_k-lr*tdw.Weight{end}.w_k;
args.Weight{end}.b_k=args.Weight{end}.b_k-lr*tdw.Weight{end}.b_k;
