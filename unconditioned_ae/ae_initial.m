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
% N=layers(2);
% W.w_k=1/M*normrnd(0,0.1,[M,N]);
% W.b_k=zeros(1,N);

layersC=layersD(2:end-1);
for i1=1:length(layersC)
    N=layersC(i1);
    W.w_c{i1}=1/N*normrnd(0,0.1,[M,N]);
    W.b_c{i1}=zeros(1,N);
end
end
function W=ed_mom_setup(layers,layersD)
M=layers(1);
% N=layers(2);
% W.w_k=zeros(M,N);
% W.b_k=zeros(1,N);

layersC=layersD(2:end-1);
for i1=1:length(layersC)
    N=layersC(i1);
    W.w_c{i1}=zeros(M,N);
    W.b_c{i1}=zeros(1,N);
end
end