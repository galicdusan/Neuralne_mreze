clear all
close all
clc

load dataset1.mat
data = data';
X = data(1:2, :);
Y = data(3, :);
X_0 = X(:, Y==0);
X_1 = X(:, Y==1);

figure(1)
hold all
scatter(X_0(1,:), X_0(2,:), 'y.')
scatter(X_1(1,:), X_1(2,:), 'c.')
xlabel('x_1')
ylabel('x_2')
title('Podaci')
legend('Klasa 0','Klasa 1')

%X_0
ind = randperm(length(X_0));
br = round(0.9*length(ind));

X_0_train = X_0(:, ind(1:br));
Y_0_train = zeros(1, br);

X_0_test = X_0(:, ind(br+1 : end));
Y_0_test = zeros(1, length(br+1:length(ind)));

%X_1
ind = randperm(length(X_1));
br = round(0.9*length(ind));

X_1_train = X_1(:, ind(1:br));
Y_1_train = ones(1, br);

X_1_test = X_1(:, ind(br+1 : end));
Y_1_test = ones(1, length(br+1:length(ind)));

%spajamo 
X_train = [X_0_train, X_1_train];
Y_train = [Y_0_train, Y_1_train];

ind = randperm(length(X_train));
X_train = X_train(:, ind);
Y_train = Y_train(:, ind);

X_test = [X_0_test, X_1_test];
Y_test = [Y_0_test, Y_1_test];

ind = randperm(length(X_test));
X_test = X_test(:, ind);
Y_test = Y_test(:, ind);

%grafik
X_train_0 = X_train(:, Y_train == 0);
X_train_1 = X_train(:, Y_train == 1);
X_test_0 = X_test(:, Y_test == 0);
X_test_1 = X_test(:, Y_test == 1);

figure(2)
hold all
scatter(X_train_0(1,:), X_train_0(2,:), 'y.')
scatter(X_test_0(1,:), X_test_0(2,:), 'bx')
xlabel('x_1')
ylabel('x_2')
title('Klasa 0')
legend('trening','test')

figure(3)
hold all
scatter(X_train_1(1,:), X_train_1(2,:), 'c.')
scatter(X_test_1(1,:), X_test_1(2,:), 'bx')
xlabel('x_1')
ylabel('x_2')
title('Klasa 1')
legend('trening','test')

struct = {[5 5], [2], [30 30 30 30]};
for j = 1:length(struct)
    structure = struct{j};

    net = patternnet(structure); 

    net.divideFcn = 'dividerand'; 

    net.divideParam.trainRatio = 1;
    net.divideParam.valRatio = 0;
    net.divideParam.testRatio = 0;

    net.performParam.regularization = 0.1;
    net.trainParam.max_fail = 37;
    net.trainParam.goal=1e-6;
    % The number of epochs for training - learn on the validation set
    net.trainParam.epochs = 500;

    %Stop when the gradient goes below this treshold
    net.trainParam.min_grad = 0;
    % Show toolbox
    net.trainparam.showWindow = true;

    net.layers{1}.transferFcn = 'tansig';
    net.layers{2}.transferFcn = 'tansig';

    [net,tr] = train(net,X_train,Y_train);
    
    %pokazujemo krivu obucavanja za finalnu mrezu
    figure()
    hold all
    plotperform(tr)
    
    %kao ulaz ubacujemo trening i test skupove u istreniranu mrezu
    Y_test_pred = sim(net, X_test);
    Y_train_pred = sim(net, X_train);

    %crtamo konfuzionu za trening, racunamo mere preciznosti
    figure()
    hold all
    title('Train set')
    plotconfusion(Y_train, Y_train_pred)

    [c,cm,ind,per] = confusion(Y_train, Y_train_pred);
    cm = cm';
    precision_train_0 = cm(1,1)/( cm(1,1)+cm(1,2) );
    recall_train_0 = cm(1,1)/( cm(1,1)+cm(2,1) );
    precision_train_1 = cm(2,2)/( cm(2,2)+cm(2,1) );
    recall_train_1 = cm(2,2)/( cm(2,2)+cm(1,2) );
  %  fscore_train = 2 * (precision_train * recall_train) / (precision_train + recall_train);
    accuracy_train = ( cm(1,1)+cm(2,2) )/( cm(1,1)+cm(2,2)+cm(1,2)+cm(2,1) );

    %crtamo konfuzionu za test, racunamo mere preciznosti
    figure()
    hold all
    title('Test set')
    plotconfusion(Y_test, Y_test_pred)

    [c,cm,ind,per] = confusion(Y_test, Y_test_pred);
    cm = cm';
    precision_test_0 = cm(1,1)/( cm(1,1)+cm(1,2) );
    recall_test_0 = cm(1,1)/( cm(1,1)+cm(2,1) );
    precision_test_1 = cm(2,2)/( cm(2,2)+cm(2,1) );
    recall_test_1 = cm(2,2)/( cm(2,2)+cm(1,2) );
 %   fscore_test = 2 * (precision_test * recall_test) / (precision_test + recall_test);
    accuracy_test = ( cm(1,1)+cm(2,2) )/( cm(1,1)+cm(2,2)+cm(1,2)+cm(2,1) );

    disp(' ')
    disp('Trenirajuci skup: ')
    disp(['Accuracy:  ' , num2str(accuracy_train)])
    disp(['precision_train_0:  ' , num2str(precision_train_0)])
    disp(['recall_train_0:  ' , num2str(recall_train_0)])
    disp(['precision_train_1:  ' , num2str(precision_train_1)])
    disp(['recall_train_1:  ' , num2str(recall_train_1)])
    disp(' ')

    disp(' ')
    disp('Testirajuci skup: ')
    disp(['Accuracy:  ' , num2str(accuracy_test)])
    disp(['precision_test_0:  ' , num2str(precision_test_0)])
    disp(['recall_test_0:  ' , num2str(recall_test_0)])
    disp(['precision_test_1:  ' , num2str(precision_test_1)])
    disp(['recall_test_1:  ' , num2str(recall_test_1)])
    disp(' ')

    %crtamo granice odlucivanja bez nesigurnosti
    xTest = [];
    opseg_y = -2:0.1:6;
    opseg_x = -4:0.1:4;
    for i = opseg_x
         xTest = [xTest [i*ones(size(opseg_y)); opseg_y]];
    end
    yTest = sim(net, xTest);
    figure()
    hold all

    Y_pred = sim(net, X);
    for i = 1:length(Y)
        if Y(1,i) ~= (Y_pred(1,i) > 0.5)
            plot (X(1,i), X(2,i), 'r.')
        end
    end

    for i = 1: length(xTest)
        if yTest(i) < 0.5
            plot (xTest(1,i), xTest(2,i), 'y.')
        elseif yTest(i) >0.5
            plot (xTest(1,i), xTest(2,i), 'c.')
        end
    end
end






