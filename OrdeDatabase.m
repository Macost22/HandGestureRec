load('DataFile1.mat')
i = 1; % frame
j = 1; % numero de sensor
k = 1; % numero de columna
feature = zeros(size(skeleton,1),139);
currentAction=1;
while(i <=size(skeleton,1))
        % Extraccion de los datos del esqueleto
        while(j <= 46)
            feature(i,k) = skeleton{i}(j,1);
            feature(i,k+1) = skeleton{i}(j,2);
            feature(i,k+2) = skeleton{i}(j,3);
            k = k+3;
            j = j+1;
        end
        j = 1;
        k = 1;
        if(i>Anotations(currentAction,2)) % Actualiza el class label para el conjunto actual
            currentAction=currentAction+1;
            classLabel=labels{currentAction,1};
        end
        % Escribir la label
        feature(i,139) = currentAction;
        i = i+1;
end
Datos1 = array2table(feature);
Datos1.Properties.VariableNames{'feature139'} = 'Clase';
%csvwrite('Datos1.csv', Datos1);
writetable(Datos1,'Test.csv');