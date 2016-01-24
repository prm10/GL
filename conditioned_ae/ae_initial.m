function args=ae_initial(args)
args.WeightEncoder=lstm_setup(args.layerEncoder);
args.Mom.WeightEncoder=lstm_mom_setup(args.layerEncoder);

args.WeightStatic=ed_setup(args.layerStatic,args.layerDecoder);
args.Mom.WeightStatic=ed_mom_setup(args.layerStatic,args.layerDecoder);

args.WeightDecoder=lstm_setup(args.layerDecoder);
args.Mom.WeightDecoder=lstm_mom_setup(args.layerDecoder);
end

function W=ed_setup(layers,layersD)
M=layers(1);
N=layers(2);
W.w_k1=1/M*normrnd(0,0.1,[M,N]);
W.b_k1=zeros(1,N);
W.w_k2=1/N*normrnd(0,0.1,[N,M]);
W.b_k2=zeros(1,M);

layersC=layersD(2:end-1);
for i1=1:length(layersC)
    M=layersC(i1);
    W.w_c{i1}=1/N*normrnd(0,0.1,[N,M]);
    W.b_c{i1}=zeros(1,M);
end
end
function W=ed_mom_setup(layers,layersD)
M=layers(1);
N=layers(2);
W.w_k1=zeros(M,N);
W.b_k1=zeros(1,N);
W.w_k2=zeros(N,M);
W.b_k2=zeros(1,M);

layersC=layersD(2:end-1);
for i1=1:length(layersC)
    M=layersC(i1);
    W.w_c{i1}=zeros(N,M);
    W.b_c{i1}=zeros(1,M);
end
end