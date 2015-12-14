function args=LSTM_weight_update(args,adw)
tdw=args.Mom;
%% 动量衰减
for i2=1:length(tdw.WeightTranR)
	tdw.WeightTranR{i2}.w_k=args.momentum*tdw.WeightTranR{i2}.w_k+adw{1}.WeightTranR{i2}.w_k*0;
	tdw.WeightTranR{i2}.b_k=args.momentum*tdw.WeightTranR{i2}.b_k+adw{1}.WeightTranR{i2}.b_k*0;
end
for i2=1:length(tdw.WeightTranP)
	tdw.WeightTranP{i2}.w_k=args.momentum*tdw.WeightTranP{i2}.w_k+adw{1}.WeightTranP{i2}.w_k*0;
	tdw.WeightTranP{i2}.b_k=args.momentum*tdw.WeightTranP{i2}.b_k+adw{1}.WeightTranP{i2}.b_k*0;
end
for i2=1:length(tdw.WeightEncoder)-1
	tdw.WeightEncoder{i2}.w_i=args.momentum*tdw.WeightEncoder{i2}.w_i+adw{1}.WeightEncoder{i2}.w_i*0;
	tdw.WeightEncoder{i2}.r_i=args.momentum*tdw.WeightEncoder{i2}.r_i+adw{1}.WeightEncoder{i2}.r_i*0;
	tdw.WeightEncoder{i2}.p_i=args.momentum*tdw.WeightEncoder{i2}.p_i+adw{1}.WeightEncoder{i2}.p_i*0;
	tdw.WeightEncoder{i2}.w_f=args.momentum*tdw.WeightEncoder{i2}.w_f+adw{1}.WeightEncoder{i2}.w_f*0;
	tdw.WeightEncoder{i2}.r_f=args.momentum*tdw.WeightEncoder{i2}.r_f+adw{1}.WeightEncoder{i2}.r_f*0;
	tdw.WeightEncoder{i2}.p_f=args.momentum*tdw.WeightEncoder{i2}.p_f+adw{1}.WeightEncoder{i2}.p_f*0;
	tdw.WeightEncoder{i2}.w_z=args.momentum*tdw.WeightEncoder{i2}.w_z+adw{1}.WeightEncoder{i2}.w_z*0;
	tdw.WeightEncoder{i2}.r_z=args.momentum*tdw.WeightEncoder{i2}.r_z+adw{1}.WeightEncoder{i2}.r_z*0;
	tdw.WeightEncoder{i2}.w_o=args.momentum*tdw.WeightEncoder{i2}.w_o+adw{1}.WeightEncoder{i2}.w_o*0;
	tdw.WeightEncoder{i2}.r_o=args.momentum*tdw.WeightEncoder{i2}.r_o+adw{1}.WeightEncoder{i2}.r_o*0;
	tdw.WeightEncoder{i2}.p_o=args.momentum*tdw.WeightEncoder{i2}.p_o+adw{1}.WeightEncoder{i2}.p_o*0;
end
tdw.WeightEncoder{end}.w_k=args.momentum*tdw.WeightEncoder{end}.w_k+adw{1}.WeightEncoder{end}.w_k*0;
tdw.WeightEncoder{end}.b_k=args.momentum*tdw.WeightEncoder{end}.b_k+adw{1}.WeightEncoder{end}.b_k*0;

for i2=1:length(tdw.WeightDecoder)-1
	tdw.WeightDecoder{i2}.w_i=args.momentum*tdw.WeightDecoder{i2}.w_i+adw{1}.WeightDecoder{i2}.w_i*0;
	tdw.WeightDecoder{i2}.r_i=args.momentum*tdw.WeightDecoder{i2}.r_i+adw{1}.WeightDecoder{i2}.r_i*0;
	tdw.WeightDecoder{i2}.p_i=args.momentum*tdw.WeightDecoder{i2}.p_i+adw{1}.WeightDecoder{i2}.p_i*0;
	tdw.WeightDecoder{i2}.w_f=args.momentum*tdw.WeightDecoder{i2}.w_f+adw{1}.WeightDecoder{i2}.w_f*0;
	tdw.WeightDecoder{i2}.r_f=args.momentum*tdw.WeightDecoder{i2}.r_f+adw{1}.WeightDecoder{i2}.r_f*0;
	tdw.WeightDecoder{i2}.p_f=args.momentum*tdw.WeightDecoder{i2}.p_f+adw{1}.WeightDecoder{i2}.p_f*0;
	tdw.WeightDecoder{i2}.w_z=args.momentum*tdw.WeightDecoder{i2}.w_z+adw{1}.WeightDecoder{i2}.w_z*0;
	tdw.WeightDecoder{i2}.r_z=args.momentum*tdw.WeightDecoder{i2}.r_z+adw{1}.WeightDecoder{i2}.r_z*0;
	tdw.WeightDecoder{i2}.w_o=args.momentum*tdw.WeightDecoder{i2}.w_o+adw{1}.WeightDecoder{i2}.w_o*0;
	tdw.WeightDecoder{i2}.r_o=args.momentum*tdw.WeightDecoder{i2}.r_o+adw{1}.WeightDecoder{i2}.r_o*0;
	tdw.WeightDecoder{i2}.p_o=args.momentum*tdw.WeightDecoder{i2}.p_o+adw{1}.WeightDecoder{i2}.p_o*0;
end
tdw.WeightDecoder{end}.w_k=args.momentum*tdw.WeightDecoder{end}.w_k+adw{1}.WeightDecoder{end}.w_k*0;
tdw.WeightDecoder{end}.b_k=args.momentum*tdw.WeightDecoder{end}.b_k+adw{1}.WeightDecoder{end}.b_k*0;

for i2=1:length(tdw.WeightPredict)-1
	tdw.WeightPredict{i2}.w_i=args.momentum*tdw.WeightPredict{i2}.w_i+adw{1}.WeightPredict{i2}.w_i*0;
	tdw.WeightPredict{i2}.r_i=args.momentum*tdw.WeightPredict{i2}.r_i+adw{1}.WeightPredict{i2}.r_i*0;
	tdw.WeightPredict{i2}.p_i=args.momentum*tdw.WeightPredict{i2}.p_i+adw{1}.WeightPredict{i2}.p_i*0;
	tdw.WeightPredict{i2}.w_f=args.momentum*tdw.WeightPredict{i2}.w_f+adw{1}.WeightPredict{i2}.w_f*0;
	tdw.WeightPredict{i2}.r_f=args.momentum*tdw.WeightPredict{i2}.r_f+adw{1}.WeightPredict{i2}.r_f*0;
	tdw.WeightPredict{i2}.p_f=args.momentum*tdw.WeightPredict{i2}.p_f+adw{1}.WeightPredict{i2}.p_f*0;
	tdw.WeightPredict{i2}.w_z=args.momentum*tdw.WeightPredict{i2}.w_z+adw{1}.WeightPredict{i2}.w_z*0;
	tdw.WeightPredict{i2}.r_z=args.momentum*tdw.WeightPredict{i2}.r_z+adw{1}.WeightPredict{i2}.r_z*0;
	tdw.WeightPredict{i2}.w_o=args.momentum*tdw.WeightPredict{i2}.w_o+adw{1}.WeightPredict{i2}.w_o*0;
	tdw.WeightPredict{i2}.r_o=args.momentum*tdw.WeightPredict{i2}.r_o+adw{1}.WeightPredict{i2}.r_o*0;
	tdw.WeightPredict{i2}.p_o=args.momentum*tdw.WeightPredict{i2}.p_o+adw{1}.WeightPredict{i2}.p_o*0;
end
tdw.WeightPredict{end}.w_k=args.momentum*tdw.WeightPredict{end}.w_k+adw{1}.WeightPredict{end}.w_k*0;
tdw.WeightPredict{end}.b_k=args.momentum*tdw.WeightPredict{end}.b_k+adw{1}.WeightPredict{end}.b_k*0;
%%  加权求和
for i1=1:length(adw)
    for i2=1:length(tdw.WeightTranR)
        tdw.WeightTranR{i2}.w_k=tdw.WeightTranR{i2}.w_k+adw{i1}.WeightTranR{i2}.w_k/length(adw);
        tdw.WeightTranR{i2}.b_k=tdw.WeightTranR{i2}.b_k+adw{i1}.WeightTranR{i2}.b_k/length(adw);
    end
    for i2=1:length(tdw.WeightTranP)
        tdw.WeightTranP{i2}.w_k=tdw.WeightTranP{i2}.w_k+adw{i1}.WeightTranP{i2}.w_k/length(adw);
        tdw.WeightTranP{i2}.b_k=tdw.WeightTranP{i2}.b_k+adw{i1}.WeightTranP{i2}.b_k/length(adw);
    end
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

    for i2=1:length(tdw.WeightPredict)-1
        tdw.WeightPredict{i2}.w_i=tdw.WeightPredict{i2}.w_i+adw{i1}.WeightPredict{i2}.w_i/length(adw);
        tdw.WeightPredict{i2}.r_i=tdw.WeightPredict{i2}.r_i+adw{i1}.WeightPredict{i2}.r_i/length(adw);
        tdw.WeightPredict{i2}.p_i=tdw.WeightPredict{i2}.p_i+adw{i1}.WeightPredict{i2}.p_i/length(adw);
        tdw.WeightPredict{i2}.w_f=tdw.WeightPredict{i2}.w_f+adw{i1}.WeightPredict{i2}.w_f/length(adw);
        tdw.WeightPredict{i2}.r_f=tdw.WeightPredict{i2}.r_f+adw{i1}.WeightPredict{i2}.r_f/length(adw);
        tdw.WeightPredict{i2}.p_f=tdw.WeightPredict{i2}.p_f+adw{i1}.WeightPredict{i2}.p_f/length(adw);
        tdw.WeightPredict{i2}.w_z=tdw.WeightPredict{i2}.w_z+adw{i1}.WeightPredict{i2}.w_z/length(adw);
        tdw.WeightPredict{i2}.r_z=tdw.WeightPredict{i2}.r_z+adw{i1}.WeightPredict{i2}.r_z/length(adw);
        tdw.WeightPredict{i2}.w_o=tdw.WeightPredict{i2}.w_o+adw{i1}.WeightPredict{i2}.w_o/length(adw);
        tdw.WeightPredict{i2}.r_o=tdw.WeightPredict{i2}.r_o+adw{i1}.WeightPredict{i2}.r_o/length(adw);
        tdw.WeightPredict{i2}.p_o=tdw.WeightPredict{i2}.p_o+adw{i1}.WeightPredict{i2}.p_o/length(adw);
    end
    tdw.WeightPredict{end}.w_k=tdw.WeightPredict{end}.w_k+adw{i1}.WeightPredict{end}.w_k/length(adw);
    tdw.WeightPredict{end}.b_k=tdw.WeightPredict{end}.b_k+adw{i1}.WeightPredict{end}.b_k/length(adw);
end
args.Mom=tdw;
%% 更新权值
lr=args.learningrate;
for i2=1:length(tdw.WeightTranR)
    args.WeightTranR{i2}.w_k=args.WeightTranR{i2}.w_k+lr*tdw.WeightTranR{i2}.w_k;
    args.WeightTranR{i2}.b_k=args.WeightTranR{i2}.b_k+lr*tdw.WeightTranR{i2}.b_k;
end
for i2=1:length(tdw.WeightTranP)
    args.WeightTranP{i2}.w_k=args.WeightTranP{i2}.w_k+lr*tdw.WeightTranP{i2}.w_k;
    args.WeightTranP{i2}.b_k=args.WeightTranP{i2}.b_k+lr*tdw.WeightTranP{i2}.b_k;
end
for i2=1:length(tdw.WeightEncoder)-1
    args.WeightEncoder{i2}.w_i=args.WeightEncoder{i2}.w_i-lr*tdw.WeightEncoder{i2}.w_i;
    args.WeightEncoder{i2}.r_i=args.WeightEncoder{i2}.r_i-lr*tdw.WeightEncoder{i2}.r_i;
    args.WeightEncoder{i2}.p_i=args.WeightEncoder{i2}.p_i-lr*tdw.WeightEncoder{i2}.p_i;
    args.WeightEncoder{i2}.w_f=args.WeightEncoder{i2}.w_f-lr*tdw.WeightEncoder{i2}.w_f;
    args.WeightEncoder{i2}.r_f=args.WeightEncoder{i2}.r_f-lr*tdw.WeightEncoder{i2}.r_f;
    args.WeightEncoder{i2}.p_f=args.WeightEncoder{i2}.p_f-lr*tdw.WeightEncoder{i2}.p_f;
    args.WeightEncoder{i2}.w_z=args.WeightEncoder{i2}.w_z-lr*tdw.WeightEncoder{i2}.w_z;
    args.WeightEncoder{i2}.r_z=args.WeightEncoder{i2}.r_z-lr*tdw.WeightEncoder{i2}.r_z;
    args.WeightEncoder{i2}.w_o=args.WeightEncoder{i2}.w_o-lr*tdw.WeightEncoder{i2}.w_o;
    args.WeightEncoder{i2}.r_o=args.WeightEncoder{i2}.r_o-lr*tdw.WeightEncoder{i2}.r_o;
    args.WeightEncoder{i2}.p_o=args.WeightEncoder{i2}.p_o-lr*tdw.WeightEncoder{i2}.p_o;
end
args.WeightEncoder{end}.w_k=args.WeightEncoder{end}.w_k-lr*tdw.WeightEncoder{end}.w_k;
args.WeightEncoder{end}.b_k=args.WeightEncoder{end}.b_k-lr*tdw.WeightEncoder{end}.b_k;
for i2=1:length(tdw.WeightDecoder)-1
    args.WeightDecoder{i2}.w_i=args.WeightDecoder{i2}.w_i-lr*tdw.WeightDecoder{i2}.w_i;
    args.WeightDecoder{i2}.r_i=args.WeightDecoder{i2}.r_i-lr*tdw.WeightDecoder{i2}.r_i;
    args.WeightDecoder{i2}.p_i=args.WeightDecoder{i2}.p_i-lr*tdw.WeightDecoder{i2}.p_i;
    args.WeightDecoder{i2}.w_f=args.WeightDecoder{i2}.w_f-lr*tdw.WeightDecoder{i2}.w_f;
    args.WeightDecoder{i2}.r_f=args.WeightDecoder{i2}.r_f-lr*tdw.WeightDecoder{i2}.r_f;
    args.WeightDecoder{i2}.p_f=args.WeightDecoder{i2}.p_f-lr*tdw.WeightDecoder{i2}.p_f;
    args.WeightDecoder{i2}.w_z=args.WeightDecoder{i2}.w_z-lr*tdw.WeightDecoder{i2}.w_z;
    args.WeightDecoder{i2}.r_z=args.WeightDecoder{i2}.r_z-lr*tdw.WeightDecoder{i2}.r_z;
    args.WeightDecoder{i2}.w_o=args.WeightDecoder{i2}.w_o-lr*tdw.WeightDecoder{i2}.w_o;
    args.WeightDecoder{i2}.r_o=args.WeightDecoder{i2}.r_o-lr*tdw.WeightDecoder{i2}.r_o;
    args.WeightDecoder{i2}.p_o=args.WeightDecoder{i2}.p_o-lr*tdw.WeightDecoder{i2}.p_o;
end
args.WeightDecoder{end}.w_k=args.WeightDecoder{end}.w_k-lr*tdw.WeightDecoder{end}.w_k;
args.WeightDecoder{end}.b_k=args.WeightDecoder{end}.b_k-lr*tdw.WeightDecoder{end}.b_k;

for i2=1:length(tdw.WeightPredict)-1
    args.WeightPredict{i2}.w_i=args.WeightPredict{i2}.w_i-lr*tdw.WeightPredict{i2}.w_i;
    args.WeightPredict{i2}.r_i=args.WeightPredict{i2}.r_i-lr*tdw.WeightPredict{i2}.r_i;
    args.WeightPredict{i2}.p_i=args.WeightPredict{i2}.p_i-lr*tdw.WeightPredict{i2}.p_i;
    args.WeightPredict{i2}.w_f=args.WeightPredict{i2}.w_f-lr*tdw.WeightPredict{i2}.w_f;
    args.WeightPredict{i2}.r_f=args.WeightPredict{i2}.r_f-lr*tdw.WeightPredict{i2}.r_f;
    args.WeightPredict{i2}.p_f=args.WeightPredict{i2}.p_f-lr*tdw.WeightPredict{i2}.p_f;
    args.WeightPredict{i2}.w_z=args.WeightPredict{i2}.w_z-lr*tdw.WeightPredict{i2}.w_z;
    args.WeightPredict{i2}.r_z=args.WeightPredict{i2}.r_z-lr*tdw.WeightPredict{i2}.r_z;
    args.WeightPredict{i2}.w_o=args.WeightPredict{i2}.w_o-lr*tdw.WeightPredict{i2}.w_o;
    args.WeightPredict{i2}.r_o=args.WeightPredict{i2}.r_o-lr*tdw.WeightPredict{i2}.r_o;
    args.WeightPredict{i2}.p_o=args.WeightPredict{i2}.p_o-lr*tdw.WeightPredict{i2}.p_o;
end
args.WeightPredict{end}.w_k=args.WeightPredict{end}.w_k-lr*tdw.WeightPredict{end}.w_k;
args.WeightPredict{end}.b_k=args.WeightPredict{end}.b_k-lr*tdw.WeightPredict{end}.b_k;


