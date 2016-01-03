function args=ae_initial(args)
args.Weight=ae_setup(args.layers);
args.Mom.Weight=ae_mom_setup(args.layers);

function Weight=ae_setup(layers)
Weight=cell(0);
for i1=1:length(layers)-1
    M=layers(i1);
    N=layers(i1+1);
    Weight{i1}.w=1/M*normrnd(0,0.1,[M,N]);
    Weight{i1}.b1=zeros(1,N);
    Weight{i1}.b2=zeros(1,M);
end

function Weight=ae_mom_setup(layers)
Weight=cell(0);
for i1=1:length(layers)-1
    M=layers(i1);
    N=layers(i1+1);
    Weight{i1}.w=zeros(M,N);
    Weight{i1}.b1=zeros(1,N);
    Weight{i1}.b2=zeros(1,M);
end