function data_P = test(test_address)

address=test_address;
temp=dir([address,'*.txt']);
temp_size=size(temp);
data_T=[];
for q=1:temp_size(1)
    cube = importdata([address,temp(q).name]);
    data_T=[data_T cube.data];
end

Fs = 2560;
T = 1/Fs;
L = 38400;             % Length of signal
t = (0:L-1)*T; 


f = Fs*(0:(L/2))/L;


fft_T=[];
for i=1:30
    Y_T=fft(data_T(:,i));
    Y_T=normalize(Y_T);
    P2=abs(Y_T/L);
    P1=P2(1:L/2+1);
    P1(2:end-1)=2*P1(2:end-1);
    fft_T=[fft_T P1];
end
fft_T=fft_T.';

data_P=[fft_T(:,328) fft_T(:,1270)];



end

