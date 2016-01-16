function adw=clamp(adw,limit)
for i1=1:length(adw)
    for i2=1:length(adw{i1}.WeightEncoder)-1
      adw{i1}.WeightEncoder{i2}.w_i=min(adw{i1}.WeightEncoder{i2}.w_i,limit*ones(size(adw{i1}.WeightEncoder{i2}.w_i)));
      adw{i1}.WeightEncoder{i2}.r_i=min(adw{i1}.WeightEncoder{i2}.r_i,limit*ones(size(adw{i1}.WeightEncoder{i2}.r_i)));
      adw{i1}.WeightEncoder{i2}.p_i=min(adw{i1}.WeightEncoder{i2}.p_i,limit*ones(size(adw{i1}.WeightEncoder{i2}.p_i)));
      adw{i1}.WeightEncoder{i2}.w_f=min(adw{i1}.WeightEncoder{i2}.w_f,limit*ones(size(adw{i1}.WeightEncoder{i2}.w_f)));
      adw{i1}.WeightEncoder{i2}.r_f=min(adw{i1}.WeightEncoder{i2}.r_f,limit*ones(size(adw{i1}.WeightEncoder{i2}.r_f)));
      adw{i1}.WeightEncoder{i2}.p_f=min(adw{i1}.WeightEncoder{i2}.p_f,limit*ones(size(adw{i1}.WeightEncoder{i2}.p_f)));
      adw{i1}.WeightEncoder{i2}.w_z=min(adw{i1}.WeightEncoder{i2}.w_z,limit*ones(size(adw{i1}.WeightEncoder{i2}.w_z)));
      adw{i1}.WeightEncoder{i2}.r_z=min(adw{i1}.WeightEncoder{i2}.r_z,limit*ones(size(adw{i1}.WeightEncoder{i2}.r_z)));
      adw{i1}.WeightEncoder{i2}.w_o=min(adw{i1}.WeightEncoder{i2}.w_o,limit*ones(size(adw{i1}.WeightEncoder{i2}.w_o)));
      adw{i1}.WeightEncoder{i2}.r_o=min(adw{i1}.WeightEncoder{i2}.r_o,limit*ones(size(adw{i1}.WeightEncoder{i2}.r_o)));
      adw{i1}.WeightEncoder{i2}.p_o=min(adw{i1}.WeightEncoder{i2}.p_o,limit*ones(size(adw{i1}.WeightEncoder{i2}.p_o)));
    end
    adw{i1}.WeightEncoder{end}.w_k=min(adw{i1}.WeightEncoder{end}.w_k,limit*ones(size(adw{i1}.WeightEncoder{end}.w_k)));
    adw{i1}.WeightEncoder{end}.b_k=min(adw{i1}.WeightEncoder{end}.b_k,limit*ones(size(adw{i1}.WeightEncoder{end}.b_k)));
	
    adw{i1}.WeightStatic.w_k1=min(adw{i1}.WeightStatic.w_k1,limit*ones(size(adw{i1}.WeightStatic.w_k1)));
    adw{i1}.WeightStatic.b_k1=min(adw{i1}.WeightStatic.b_k1,limit*ones(size(adw{i1}.WeightStatic.b_k1)));
    adw{i1}.WeightStatic.w_k2=min(adw{i1}.WeightStatic.w_k2,limit*ones(size(adw{i1}.WeightStatic.w_k2)));
    adw{i1}.WeightStatic.b_k2=min(adw{i1}.WeightStatic.b_k2,limit*ones(size(adw{i1}.WeightStatic.b_k2)));
	
    for i2=1:length(adw{i1}.WeightDecoder)-1
      adw{i1}.WeightDecoder{i2}.w_i=min(adw{i1}.WeightDecoder{i2}.w_i,limit*ones(size(adw{i1}.WeightDecoder{i2}.w_i)));
      adw{i1}.WeightDecoder{i2}.r_i=min(adw{i1}.WeightDecoder{i2}.r_i,limit*ones(size(adw{i1}.WeightDecoder{i2}.r_i)));
      adw{i1}.WeightDecoder{i2}.p_i=min(adw{i1}.WeightDecoder{i2}.p_i,limit*ones(size(adw{i1}.WeightDecoder{i2}.p_i)));
      adw{i1}.WeightDecoder{i2}.w_f=min(adw{i1}.WeightDecoder{i2}.w_f,limit*ones(size(adw{i1}.WeightDecoder{i2}.w_f)));
      adw{i1}.WeightDecoder{i2}.r_f=min(adw{i1}.WeightDecoder{i2}.r_f,limit*ones(size(adw{i1}.WeightDecoder{i2}.r_f)));
      adw{i1}.WeightDecoder{i2}.p_f=min(adw{i1}.WeightDecoder{i2}.p_f,limit*ones(size(adw{i1}.WeightDecoder{i2}.p_f)));
      adw{i1}.WeightDecoder{i2}.w_z=min(adw{i1}.WeightDecoder{i2}.w_z,limit*ones(size(adw{i1}.WeightDecoder{i2}.w_z)));
      adw{i1}.WeightDecoder{i2}.r_z=min(adw{i1}.WeightDecoder{i2}.r_z,limit*ones(size(adw{i1}.WeightDecoder{i2}.r_z)));
      adw{i1}.WeightDecoder{i2}.w_o=min(adw{i1}.WeightDecoder{i2}.w_o,limit*ones(size(adw{i1}.WeightDecoder{i2}.w_o)));
      adw{i1}.WeightDecoder{i2}.r_o=min(adw{i1}.WeightDecoder{i2}.r_o,limit*ones(size(adw{i1}.WeightDecoder{i2}.r_o)));
      adw{i1}.WeightDecoder{i2}.p_o=min(adw{i1}.WeightDecoder{i2}.p_o,limit*ones(size(adw{i1}.WeightDecoder{i2}.p_o)));
    end
    adw{i1}.WeightDecoder{end}.w_k=min(adw{i1}.WeightDecoder{end}.w_k,limit*ones(size(adw{i1}.WeightDecoder{end}.w_k)));
    adw{i1}.WeightDecoder{end}.b_k=min(adw{i1}.WeightDecoder{end}.b_k,limit*ones(size(adw{i1}.WeightDecoder{end}.b_k)));
end

for i1=1:length(adw)
    for i2=1:length(adw{i1}.WeightEncoder)-1
      adw{i1}.WeightEncoder{i2}.w_i=max(adw{i1}.WeightEncoder{i2}.w_i,-limit*ones(size(adw{i1}.WeightEncoder{i2}.w_i)));
      adw{i1}.WeightEncoder{i2}.r_i=max(adw{i1}.WeightEncoder{i2}.r_i,-limit*ones(size(adw{i1}.WeightEncoder{i2}.r_i)));
      adw{i1}.WeightEncoder{i2}.p_i=max(adw{i1}.WeightEncoder{i2}.p_i,-limit*ones(size(adw{i1}.WeightEncoder{i2}.p_i)));
      adw{i1}.WeightEncoder{i2}.w_f=max(adw{i1}.WeightEncoder{i2}.w_f,-limit*ones(size(adw{i1}.WeightEncoder{i2}.w_f)));
      adw{i1}.WeightEncoder{i2}.r_f=max(adw{i1}.WeightEncoder{i2}.r_f,-limit*ones(size(adw{i1}.WeightEncoder{i2}.r_f)));
      adw{i1}.WeightEncoder{i2}.p_f=max(adw{i1}.WeightEncoder{i2}.p_f,-limit*ones(size(adw{i1}.WeightEncoder{i2}.p_f)));
      adw{i1}.WeightEncoder{i2}.w_z=max(adw{i1}.WeightEncoder{i2}.w_z,-limit*ones(size(adw{i1}.WeightEncoder{i2}.w_z)));
      adw{i1}.WeightEncoder{i2}.r_z=max(adw{i1}.WeightEncoder{i2}.r_z,-limit*ones(size(adw{i1}.WeightEncoder{i2}.r_z)));
      adw{i1}.WeightEncoder{i2}.w_o=max(adw{i1}.WeightEncoder{i2}.w_o,-limit*ones(size(adw{i1}.WeightEncoder{i2}.w_o)));
      adw{i1}.WeightEncoder{i2}.r_o=max(adw{i1}.WeightEncoder{i2}.r_o,-limit*ones(size(adw{i1}.WeightEncoder{i2}.r_o)));
      adw{i1}.WeightEncoder{i2}.p_o=max(adw{i1}.WeightEncoder{i2}.p_o,-limit*ones(size(adw{i1}.WeightEncoder{i2}.p_o)));
    end
    adw{i1}.WeightEncoder{end}.w_k=max(adw{i1}.WeightEncoder{end}.w_k,-limit*ones(size(adw{i1}.WeightEncoder{end}.w_k)));
    adw{i1}.WeightEncoder{end}.b_k=max(adw{i1}.WeightEncoder{end}.b_k,-limit*ones(size(adw{i1}.WeightEncoder{end}.b_k)));
	
    adw{i1}.WeightStatic.w_k1=max(adw{i1}.WeightStatic.w_k1,-limit*ones(size(adw{i1}.WeightStatic.w_k1)));
    adw{i1}.WeightStatic.b_k1=max(adw{i1}.WeightStatic.b_k1,-limit*ones(size(adw{i1}.WeightStatic.b_k1)));
    adw{i1}.WeightStatic.w_k2=max(adw{i1}.WeightStatic.w_k2,-limit*ones(size(adw{i1}.WeightStatic.w_k2)));
    adw{i1}.WeightStatic.b_k2=max(adw{i1}.WeightStatic.b_k2,-limit*ones(size(adw{i1}.WeightStatic.b_k2)));
	
    for i2=1:length(adw{i1}.WeightDecoder)-1
      adw{i1}.WeightDecoder{i2}.w_i=max(adw{i1}.WeightDecoder{i2}.w_i,-limit*ones(size(adw{i1}.WeightDecoder{i2}.w_i)));
      adw{i1}.WeightDecoder{i2}.r_i=max(adw{i1}.WeightDecoder{i2}.r_i,-limit*ones(size(adw{i1}.WeightDecoder{i2}.r_i)));
      adw{i1}.WeightDecoder{i2}.p_i=max(adw{i1}.WeightDecoder{i2}.p_i,-limit*ones(size(adw{i1}.WeightDecoder{i2}.p_i)));
      adw{i1}.WeightDecoder{i2}.w_f=max(adw{i1}.WeightDecoder{i2}.w_f,-limit*ones(size(adw{i1}.WeightDecoder{i2}.w_f)));
      adw{i1}.WeightDecoder{i2}.r_f=max(adw{i1}.WeightDecoder{i2}.r_f,-limit*ones(size(adw{i1}.WeightDecoder{i2}.r_f)));
      adw{i1}.WeightDecoder{i2}.p_f=max(adw{i1}.WeightDecoder{i2}.p_f,-limit*ones(size(adw{i1}.WeightDecoder{i2}.p_f)));
      adw{i1}.WeightDecoder{i2}.w_z=max(adw{i1}.WeightDecoder{i2}.w_z,-limit*ones(size(adw{i1}.WeightDecoder{i2}.w_z)));
      adw{i1}.WeightDecoder{i2}.r_z=max(adw{i1}.WeightDecoder{i2}.r_z,-limit*ones(size(adw{i1}.WeightDecoder{i2}.r_z)));
      adw{i1}.WeightDecoder{i2}.w_o=max(adw{i1}.WeightDecoder{i2}.w_o,-limit*ones(size(adw{i1}.WeightDecoder{i2}.w_o)));
      adw{i1}.WeightDecoder{i2}.r_o=max(adw{i1}.WeightDecoder{i2}.r_o,-limit*ones(size(adw{i1}.WeightDecoder{i2}.r_o)));
      adw{i1}.WeightDecoder{i2}.p_o=max(adw{i1}.WeightDecoder{i2}.p_o,-limit*ones(size(adw{i1}.WeightDecoder{i2}.p_o)));
    end
    adw{i1}.WeightDecoder{end}.w_k=max(adw{i1}.WeightDecoder{end}.w_k,-limit*ones(size(adw{i1}.WeightDecoder{end}.w_k)));
    adw{i1}.WeightDecoder{end}.b_k=max(adw{i1}.WeightDecoder{end}.b_k,-limit*ones(size(adw{i1}.WeightDecoder{end}.b_k)));
end