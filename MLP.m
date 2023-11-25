% NN Multilayer Perceptron

clear all
close all
clc

Nhidd = 10;
Repts = 100;
Fol = 5; 

for Sujetos = 1:30
    
    clearvars -except Metrics Sujetos Nhidd Repts Fol

    % Load Data
    load(['..\Features\PSD',num2str(Sujetos),'.mat'])
    
    for Runs1 = 1:5

        % Define features and classes matrix to train 
        Features = [DataPSD(Runs1).cross; DataPSD(Runs1).right];
        Class = [ones(size(DataPSD(Runs1).cross,1),1)*1; ones(size(DataPSD(Runs1).right,1),1)*2];
        Label = Class;
        
        % Cross Validation
        hpartition = cvpartition(length(Features),'kFold',Fol);
    
        % CONFIGURATION AND TRAINING OF NEURAL NETWORK
        for Hidd = 1:Nhidd
            Acc_base = 0;
            for Inic = 1:Repts
                Start_opt1 = tic; 
                for kfold = 1:Fol
                    
                    % Data
                    xtrain = Features(~hpartition.test(kfold),:);
                    ytrain = Label(~hpartition.test(kfold),:);
                    xtest = Features(hpartition.test(kfold),:);
                    ytest = Label(hpartition.test(kfold),:);
                    
                    % Adjust data
                    xtest = (xtest - mean(xtrain)) ./ std(xtrain); % Normalized Z-score
                    xtrain = (xtrain - mean(xtrain)) ./ std(xtrain); % Normalized Z-score
                    
                    % Parameters
                    tstart_opt2_train = tic;
                    input_layer_size  = size(xtrain,2); % Input layer size correspond to the size of the used features 
                    hidden_layer_size = Hidd; % Hidden layer can change
                    num_labels = max(ytrain); % The output layer correspond to the predicted class, for this case are 2 classes

                    % Inizializing parameters
                    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size); % Theta 1 and Theta 2 are initialized of random form between values of -epsilon/2 and epsilon/2
                    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
                    
                    initial_nn_params = [initial_Theta1(:); initial_Theta2(:)];
            
                    options = optimset('MaxIter', 10); % The maximum number of iterations is established at 10
            
                    lambda = 0; % Regulation parameter have a value of zero
            
                    costFunction = @(p) nnCostFunction(p, ...
                                                       input_layer_size, ...
                                                       hidden_layer_size, ...
                                                       num_labels, xtrain, ytrain, lambda); % The cost function is calculated, also the gradient and the error matrix that allow to updating the Theta1 and Theta2 Values 
            
                    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options); % This function update the Theta1 and Theta2 values that generate the minimum cost (local minimum)
            
                    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                                     hidden_layer_size, (input_layer_size + 1));
            
                    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                                     num_labels, (hidden_layer_size + 1)); 
                    net.T1 = Theta1;
                    net.T2 = Theta2;
                    tend_opt2_train = toc(tstart_opt2_train);

                    % Predict the testing values
                    pred_train = predict(Theta1, Theta2, xtrain);
                    Acc_tr = sum(ytrain==pred_train)/size(pred_train,1);
                    cc_tr = confusionmat(ytrain,pred_train);
                    Fpr_tr = cc_tr(2,1)/(cc_tr(2,1)+cc_tr(2,2));

                    % Predict the testing values
                    tstart_opt2_test = tic;
                    pred_test = predict(Theta1, Theta2, xtest);
                    tend_opt2_test = toc(tstart_opt2_test);
                    Acc_te = sum(ytest==pred_test)/size(pred_test,1);
                    cc_te = confusionmat(ytest,pred_test);
                    Fpr_te = cc_te(2,1)/(cc_te(2,1)+cc_te(2,2));

                    % Guardamos las mejores redes
                    if Acc_te >= Acc_base 
                          r = Inic;
                          Acc_base = Acc_te;
                          red = net;
                    end
                    % Guardamos los resultados
                    Acc_kfold(kfold,:) = [Acc_tr Acc_te];
                    Fpr_kfold(kfold,:) = [Fpr_tr Fpr_te];
                    Time_kfold(kfold,:) = [tend_opt2_train tend_opt2_test];
                    Cc_kfold(:,:,kfold) = cc_te;
                end
                End_opt1(Inic,:) = toc(Start_opt1);
                Acc_Increment(Inic,:) = mean(Acc_kfold);
                Fpr_Increment(Inic,:) = mean(Fpr_kfold);
                Time_Increment(Inic,:) = mean(Time_kfold);
                Cc_Increment(:,:,Inic) = sum(Cc_kfold,3);
            end
            Time1Opt(:,Hidd) = End_opt1;
            Result(Runs1).AccHidd{Hidd} = Acc_Increment;
            Result(Runs1).FprHidd{Hidd} = Fpr_Increment;
            Result(Runs1).TimHidd{Hidd} = Time_Increment;
            Result(Runs1).CcoHidd{Hidd} = Cc_Increment;
            Result(Runs1).NetHidd(Hidd).net = red;
        end
        
        clear Theta1 Theta2 initial_Theta1 initial_Theta2 options costFunction nn_params cost ...
              xtrain ytrain xtest ytrain Features Class Label

        % Se evalua cual es el mejor modelo
        for h = 1:Nhidd
            Val = Result(Runs1).AccHidd{h};
            Best_tr(h) = max(Val(:,1));
            Best_ts(h) = max(Val(:,2));
        end
        Best_better_tr = find(Best_tr == max(Best_tr));
        Best_better_ts = find(Best_ts == max(Best_ts));
        red_tr = Result(Runs1).NetHidd(Best_better_tr).net;
        red_ts = Result(Runs1).NetHidd(Best_better_ts).net;
        Theta_RT1 = red_ts.T1; 
        Theta_RT2 = red_ts.T2;
        [~,hidd] = max(Best_ts);
        Count = 1;
        
        % Retraining
        for Runs2 = 1:5
            if Runs2~=Runs1
                
               % Define features and classes matrix to train 
               Features = [DataPSD(Runs2).cross; DataPSD(Runs2).down];
               Class = [ones(size(DataPSD(Runs2).cross,1),1)*1; ones(size(DataPSD(Runs2).down,1),1)*2];
               Label = Class;
               
               for kfold = 1:Fol
                    
                   % Data
                   xtrain = Features(~hpartition.test(kfold),:);
                   ytrain = Label(~hpartition.test(kfold),:);
                   xtest = Features(hpartition.test(kfold),:);
                   ytest = Label(hpartition.test(kfold),:);
                    
                   % Adjust data
                   xtest = (xtest - mean(xtrain))./std(xtrain); % Normalized Z-score
                   xtrain = (xtrain - mean(xtrain))./std(xtrain); % Normalized Z-score

                   % Predict the testing values without retraining
                   tstart_ant_test = tic;
                   pred_test_ant = predict(red_ts.T1, red_ts.T2, xtest);
                   tend_ant_test(kfold,:) = toc(tstart_ant_test);
                   Acc_ant_te(kfold) = sum(ytest==pred_test_ant)/size(pred_test_ant,1);
                   cc_ant_te(:,:,kfold) = confusionmat(ytest,pred_test_ant);
                   Fpr_ant_te(kfold) = cc_ant_te(2,1)/(cc_ant_te(2,1)+cc_ant_te(2,2));
                   
                   Start_inc1 = tic;
                   for Ins = 1:Repts
                        
                        % Parameters
                        input_layer_size  = size(xtrain,2); % Input layer size correspond to the size of the used features 
                        hidden_layer_size = hidd; % Hidden layer can change, 
                        num_labels = max(ytrain); % The output layer correspond to the predicted class, for this case are 2 classes
                        
                        % Inizializing parameters
                        Start_inc2_train = tic;
                        initial_nn_params = [Theta_RT1(:); Theta_RT2(:)];
                        
                        options = optimset('MaxIter', 10); % The maximum number of iterations is established at 50
                
                        lambda = 0; % Regulation parameter have a value of zero
                
                        costFunction = @(p) nnCostFunction(p, ...
                                                           input_layer_size, ...
                                                           hidden_layer_size, ...
                                                           num_labels, xtrain, ytrain, lambda); % The cost function is calculated, also the gradient and the error matrix that allow to updating the Theta1 and Theta2 Values 
                
                        [nn_params, cost] = fmincg(costFunction, initial_nn_params, options); % This function update the Theta1 and Theta2 values that generate the minimum cost (local minimum)
                
                        Theta_RT1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                                         hidden_layer_size, (input_layer_size + 1));
                
                        Theta_RT2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                                         num_labels, (hidden_layer_size + 1));
                        End_inc2_train(Ins,:) = toc(Start_inc2_train);

                        % Predict the testing values
                        Start_inc2_test = tic;
                        pred_test_con = predict(Theta_RT1, Theta_RT2, xtest);
                        End_inc2_test(Ins,:) = toc(Start_inc2_test);
                        Acc_con_te(Ins,:) = sum(ytest==pred_test_con)/size(pred_test_con,1);
                        cc_con_te = confusionmat(ytest,pred_test_con);
                        Fpr_con_te(Ins,:) = cc_con_te(2,1)/(cc_con_te(2,1)+cc_con_te(2,2));
                        Cc_inc(:,:,Ins) = cc_con_te;
                   end
                   End_inc1(:,kfold) = toc(Start_inc1);
                   End_inc2Final_tr(:,kfold) = End_inc2_train(end,:);
                   End_inc2Final_ts(:,kfold) = End_inc2_test(end,:);
                   Acc_rtraining(:,kfold) = Acc_con_te(end,:);
                   Fpr_rtraining(:,kfold) = Fpr_con_te(end,:);
                   Cc_rtraining(:,:,kfold) = Cc_inc(:,:,end);
%                    Theta_RT1 = red_ts.T1; 
%                    Theta_RT2 = red_ts.T2;
               end
               
               Retrining(Count).ConTime1 = mean(End_inc1);
               Retrining(Count).ConTime2_tr = mean(End_inc2Final_tr,2);
               Retrining(Count).ConTime2_ts = mean(End_inc2Final_ts,2);
               Retrining(Count).ConAcc_ins = mean(Acc_rtraining,2);
               Retrining(Count).ConFpr_ins = mean(Fpr_rtraining,2);
               Retrining(Count).ConCc_ins = sum(Cc_rtraining,3);
               Retrining(Count).SinCc_ins = sum(cc_ant_te,3);
               Retrining(Count).SinTime_ts = tend_ant_test;
               Retrining(Count).SinAcc_ins = mean(Acc_ant_te,2);
               Retrining(Count).SinFpr_ins = mean(Fpr_ant_te,2);
               Count = Count + 1;
            end
        end
        ResultsHiddFin = Result;
        RetriningFin{Runs1} = Retrining;
    end
    display(['Sujeto ',num2str(Sujetos)])
    Metrics.Retrain = RetriningFin;
    Metrics.Normal = ResultsHiddFin;
    save(['Result_Sub',num2str(Sujetos)],'Metrics')
end

