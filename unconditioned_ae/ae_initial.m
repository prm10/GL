function args=ae_initial(args)
  args.WeightEncoder=lstm_setup(args.layerEncoder);
  args.Mom.WeightEncoder=lstm_mom_setup(args.layerEncoder);

  args.WeightStatic=ed_setup(args.layerStatic);
  args.Mom.WeightStatic=ed_mom_setup(args.layerStatic,args.layerDecoder);

  args.WeightDecoder=lstm_setup(args.layerDecoder);
  args.Mom.WeightDecoder=lstm_mom_setup(args.layerDecoder);
end

function W=ed_setup(layers)
  M=layers(1);
  N=layers(2);
  W.w_c=1/N*normrnd(0,0.01,[M,N]);
  W.b_c=zeros(1,N);
end

function W=ed_mom_setup(layers,layersD)
  M=layers(1);
  N=layers(2);
  W.w_c=zeros(M,N);
  W.b_c=zeros(1,N);
end
