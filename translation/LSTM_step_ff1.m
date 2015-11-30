function [xin,in2,f2,z2,c,o2,yout]=LSTM_step_ff1(x0,y0,c0,Weight)
    xin=[x0 1];
    % input gates
    in1=xin*Weight.w_i+y0*Weight.r_i+c0.*Weight.p_i;
    in2=sigmoid(in1);
    % forget gates
    f1=xin*Weight.w_f+y0*Weight.r_f+c0.*Weight.p_f;
    f2=sigmoid(f1(t,:));
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
