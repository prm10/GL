function [x2,in2,f2,z2,c2,o2,y2,output]=LSTM_step_ff1(input_x11,input_c0,input_xC,Weight,T,outputLayer)
    if(size(input_x11,2)==size(input_xC,2))%编码,input_c0为0，input_xC为输入
        inputX0=input_xC;
        w_k=Weight{end}.w_k;
        b_k=Weight{end}.b_k;
        for t=1:T
            inputX=inputX0(t,:);
            for lay_i=1:length(Weight)-1
                if t==1
                    inputY=zeros(1,size(Weight{lay_i}.r_i,2));
                    inputC=zeros(1,size(Weight{lay_i}.r_i,2));
                else
                    inputY=y2{lay_i}(t-1,:);
                    inputC=c2{lay_i}(t-1,:);
                end
                [x2{lay_i}(t,:),in2{lay_i}(t,:),f2{lay_i}(t,:),z2{lay_i}(t,:),c2{lay_i}(t,:),o2{lay_i}(t,:),y2{lay_i}(t,:)]=LSTM_step_ff2(inputX,inputY,inputC,Weight{lay_i});
                inputX=y2{lay_i}(t,:);
            end
            %最后一层
            z_k2=inputX*w_k+b_k;
            switch outputLayer
                case{'softmax'}
                    temp=exp(z_k2-max(z_k2));
                    output(t,:)=temp/sum(temp);
                otherwise
                    output(t,:)=tanh(z_k2);
            end
        end
    else%解码或预测,input_xC为C1
        inputX0(1,:)=input_x11;%[C1,zeros(1,size(input,2))];
        w_k=Weight{end}.w_k;
        b_k=Weight{end}.b_k;
        for t=1:T
            inputX=inputX0(t,:);
            for lay_i=1:length(Weight)-1
                if t==1
                    inputY=zeros(1,size(Weight{lay_i}.r_i,2));
                    inputC=input_c0{lay_i};%zeros(1,args.decoderLayer(lay_i+1));
                else
                    inputY=y2{lay_i}(t-1,:);
                    inputC=c2{lay_i}(t-1,:);
                end
                [x2{lay_i}(t,:),in2{lay_i}(t,:),f2{lay_i}(t,:),z2{lay_i}(t,:),c2{lay_i}(t,:),o2{lay_i}(t,:),y2{lay_i}(t,:)]=LSTM_step_ff2(inputX,inputY,inputC,Weight{lay_i});
                inputX=y2{lay_i}(t,:);
            end
            %最后一层
            z_k2=inputX*w_k+b_k;
            inputX0(t+1,:)=[input_xC,tanh(z_k2)];
        end
        output=inputX0(2:end,size(input_xC,2)+1:end);
    end
    
function [xin,in2,f2,z2,c,o2,yout]=LSTM_step_ff2(x1,y0,c0,Weight)
    xin=[x1,1];
    % input gates
    in1=xin*Weight.w_i+y0*Weight.r_i+c0.*Weight.p_i;
    in2=sigmoid(in1);
    % forget gates
    f1=xin*Weight.w_f+y0*Weight.r_f+c0.*Weight.p_f;
    f2=sigmoid(f1);
    % cells
    z1=xin*Weight.w_z+y0*Weight.r_z;
    z2=tanh(z1);
    c=f2.*c0+in2.*z2;
    % output gates
    o1=xin*Weight.w_o+y0*Weight.r_o+c.*Weight.p_o;
    o2=sigmoid(o1);
    yout=o2.*tanh(c);

function y=sigmoid(x)
    y=1./(1+exp(-x));
