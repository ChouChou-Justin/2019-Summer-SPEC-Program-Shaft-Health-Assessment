clear;close all
address='C:\Users\Justin\Desktop\SPEC Homework2\Training\Healthy';
temp=dir([address,'*.txt']);
temp_size=size(temp);
data_H=[];
for q=1:temp_size(1)
    cube = importdata([address,temp(q).name]);
    data_H=[data_H cube.data];
end

mean_H=mean(data_H);
var_H=var(data_H);
skew_H=skewness(data_H);
kurts_H=kurtosis(data_H);



address='C:\Users\Justin\Desktop\SPEC Homework2\Training\Faulty';
temp=dir([address,'*.txt']);
temp_size=size(temp);
data_F=[];
for q=1:temp_size(1)
    cube = importdata([address,temp(q).name]);
    data_F=[data_F cube.data];
end
mean_F=mean(data_F);
var_F=var(data_F);
skew_F=skewness(data_F);
kurts_F=kurtosis(data_F);

% figure;
% plot(mean_H)
% hold on
% plot(mean_F)
% title('Mean')
% 
% figure;
% plot(var_H)
% hold on
% plot(var_F)
% title('variance')
% 
% figure;
% plot(skew_H)
% hold on
% plot(skew_F)
% title('skewness')
% 
% figure;
% plot(kurts_H)
% hold on
% plot(kurts_F)
% title('kurtosis')

Fs = 2560;
T = 1/Fs;
L = 38400;             % Length of signal
t = (0:L-1)*T; 

% Y_H=fft(data_H(:,1));
% 
% P2 = abs(Y_H/L);
% P1 = P2(1:L/2+1);
% P1(2:end-1) = 2*P1(2:end-1);
% 
 f = Fs*(0:(L/2))/L;
% plot(f,P1) 
% title('Single-Sided Amplitude Spectrum of X(t)')
% xlabel('f (Hz)')
% ylabel('|P1(f)|')

fft_H=[];
for i=1:20
    Y_H=fft(data_H(:,i));
    Y_H=normalize(Y_H);
    P2=abs(Y_H/L);
    P1=P2(1:L/2+1);
    P1(2:end-1)=2*P1(2:end-1);
    fft_H=[fft_H P1];
end

fft_F=[];
for i=1:20
    Y_F=fft(data_F(:,i));
    Y_F=normalize(Y_F);
    P2=abs(Y_F/L);
    P1=P2(1:L/2+1);
    P1(2:end-1)=2*P1(2:end-1);
    fft_F=[fft_F P1];
end

fft_F=fft_F.';
fft_H=fft_H.';
figure;plot(f,fft_H(1,:));
hold on
plot(f,fft_F(1,:));
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')
legend('Health','Faulty')

for i=1:length(fft_H)
        inter=abs(mean(nonzeros(fft_H(:,i)))-mean(nonzeros(fft_F(:,i))));
        intra=std(nonzeros(fft_H(:,i)))+std(nonzeros(fft_F(:,i)));
        SI(i)=inter/intra;
end

figure;plot(f,SI)
[sort_V loc]=sort(SI,'descend');

xlabel('Hz')
ylabel('score')
title('Fisher Criterion');

training=[fft_H(:,328) fft_H(:,1270) ones(20,1)*0.95;fft_F(:,328) fft_F(:,1270) ones(20,1)*0.05];
training=training(randsample(1:length(training),length(training)),:);

% for i=1:40
%     if training(i,3)==0
%         label(i)={'H'};
%     elseif training(i,3)==1
%         label(i)={'F'};
%     end
% end
% 
% 
% label=label.';
% label=categorical(label);
label=training(:,3);
%%
train1_subset=training(11:40,1:2);
label1_train=label(11:40);
test1_subset=training(1:10,1:2);
label1_test=label(1:10);

beta1 = glmfit(train1_subset,label1_train,'binomial');
CV_Test_1 = glmval(beta1,test1_subset,'logit') ;  %Use LR Model
id1=[];
ROC_H1=[];
ROC_F1=[];
for i=1:10
    if CV_Test_1(i)>0.5
       id1(i,1)=0.95;
       ROC_H1=[ROC_H1;test1_subset(i,1:2)];
    else
       id1(i,1)=0.05;
       ROC_F1=[ROC_F1;test1_subset(i,1:2)];
    end
end
figure
roc_curve(ROC_H1,ROC_F1);



acc1=round(id1)==round(training(1:10,3));
figure
plotconfusion(round(id1)',round(training(1:10,3))');

%%
train2_subset=[training(1:10,1:2);training(21:40,1:2)];
label2_train=[label(1:10);label(21:40)];
test2_subset=training(11:20,1:2);
label2_test=label(11:20);


beta2 = glmfit(train2_subset,label2_train,'binomial');
CV_Test_2 = glmval(beta2,test2_subset,'logit') ;  %Use LR Model
id2=[];
ROC_H2=[];
ROC_F2=[];
for i=1:10
    if CV_Test_2(i)>0.5
       id2(i,1)=0.95;
       ROC_H2=[ROC_H2;test2_subset(i,1:2)];
    else
       id2(i,1)=0.05;
       ROC_F2=[ROC_F2;test2_subset(i,1:2)];
    end
end
figure
roc_curve(ROC_H2,ROC_F2);

acc2=round(id2)==round(training(11:20,3));
figure
plotconfusion(round(id2)',round(training(11:20,3))');


%%
train3_subset=[training(1:20,1:2);training(31:40,1:2)];
label3_train=[label(1:20);label(31:40)];
test3_subset=training(21:30,1:2);
label3_test=label(21:30);


beta3 = glmfit(train3_subset,label3_train,'binomial');
CV_Test_3 = glmval(beta3,test3_subset,'logit') ;  %Use LR Model
id3=[];
ROC_H3=[];
ROC_F3=[];
for i=1:10
    if CV_Test_3(i)>0.5
       id3(i,1)=0.95;
       ROC_H3=[ROC_H3;test3_subset(i,1:2)];
    else
       id3(i,1)=0.05;
       ROC_F3=[ROC_F3;test3_subset(i,1:2)];
    end
end
figure
roc_curve(ROC_H3,ROC_F3);

acc3=round(id3)==round(training(21:30,3));
figure
plotconfusion(round(id3)',round(training(21:30,3))');

%%

train4_subset=training(1:30,1:2);
label4_train=label(1:30);
test4_subset=training(31:40,1:2);
label4_test=label(31:40);

beta4 = glmfit(train4_subset,label4_train,'binomial');
CV_Test_4 = glmval(beta4,test4_subset,'logit') ;  %Use LR Model
id4=[];
ROC_H4=[];
ROC_F4=[];
for i=1:10
    if CV_Test_4(i)>0.5
       id4(i,1)=0.95;
       ROC_H4=[ROC_H4;test4_subset(i,1:2)];
    else
       id4(i,1)=0.05;
       ROC_F4=[ROC_F4;test4_subset(i,1:2)];
    end
end
figure
roc_curve(ROC_H4,ROC_F4);

acc4=round(id4)==round(training(31:40,3));
figure
plotconfusion(round(id4)',round(training(31:40,3))');

%%
acc_total=(sum(acc1)+sum(acc2)+sum(acc3)+sum(acc4))/40;


% The column vector, species, consists of iris flowers of three different species, setosa, versicolor, virginica. The double matrix meas consists of four types of measurements on the flowers, the length and width of sepals and petals in centimeters, respectively.
% Define the nominal response variable using a categorical array.
% Fit a multinomial regression model to predict the species using the measurements.  


testing=test('C:\Users\Justin\Desktop\SPEC Homework2\Testing\');
%%
%1
CV_Test_final1 = glmval(beta1,testing,'logit') ;  %Use LR Model
figure;
plot(CV_Test_final1)
xlabel('# samples');
ylabel('CV');


%2
CV_Test_final2 = glmval(beta2,testing,'logit') ;  %Use LR Model
figure;
plot(CV_Test_final2)
xlabel('# samples');
ylabel('CV');

%3
CV_Test_final3 = glmval(beta3,testing,'logit') ;  %Use LR Model
figure;
plot(CV_Test_final3)
xlabel('# samples');
ylabel('CV');
%4
CV_Test_final4 = glmval(beta4,testing,'logit') ;  %Use LR Model
figure;
plot(CV_Test_final4)
xlabel('# samples');
ylabel('CV');











