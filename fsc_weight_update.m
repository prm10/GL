function args=fsc_weight_update(args,adw)
tdw=args.Mom;%total gradient weight
%% 动量衰减
wd=args.weightDecay;
for i2=1:length(tdw.Weight)-1
	tdw.Weight{i2}.w_i=args.momentum*tdw.Weight{i2}.w_i-wd*args.Weight{i2}.w_i;
	tdw.Weight{i2}.r_i=args.momentum*tdw.Weight{i2}.r_i-wd*args.Weight{i2}.r_i;
	tdw.Weight{i2}.p_i=args.momentum*tdw.Weight{i2}.p_i-wd*args.Weight{i2}.p_i;
	tdw.Weight{i2}.w_f=args.momentum*tdw.Weight{i2}.w_f-wd*args.Weight{i2}.w_f;
	tdw.Weight{i2}.r_f=args.momentum*tdw.Weight{i2}.r_f-wd*args.Weight{i2}.r_f;
	tdw.Weight{i2}.p_f=args.momentum*tdw.Weight{i2}.p_f-wd*args.Weight{i2}.p_f;
	tdw.Weight{i2}.w_z=args.momentum*tdw.Weight{i2}.w_z-wd*args.Weight{i2}.w_z;
	tdw.Weight{i2}.r_z=args.momentum*tdw.Weight{i2}.r_z-wd*args.Weight{i2}.r_z;
	tdw.Weight{i2}.w_o=args.momentum*tdw.Weight{i2}.w_o-wd*args.Weight{i2}.w_o;
	tdw.Weight{i2}.r_o=args.momentum*tdw.Weight{i2}.r_o-wd*args.Weight{i2}.r_o;
	tdw.Weight{i2}.p_o=args.momentum*tdw.Weight{i2}.p_o-wd*args.Weight{i2}.p_o;
end
tdw.Weight{end}.w_k=args.momentum*tdw.Weight{end}.w_k-wd*args.Weight{end}.w_k;
tdw.Weight{end}.b_k=args.momentum*tdw.Weight{end}.b_k-wd*args.Weight{end}.b_k;
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

