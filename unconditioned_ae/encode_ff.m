function output=encode_ff(args,input)
x1=input;
for i1=1:length(args.layerEncoder)-2
    c0=zeros(1,size(args.WeightEncoder{i1}.r_i,1));
    [~,~,~,~,~,~,y2]=LSTM_step_ff_fast(x1,c0,args.WeightEncoder{i1});
    x1=y2;
end
cEncoder=tanh_output_ff(args.WeightEncoder{end}.w_k,args.WeightEncoder{end}.b_k,y2);
output=tanh_output_ff(args.WeightStatic.w_k1,args.WeightStatic.b_k1,cEncoder);

