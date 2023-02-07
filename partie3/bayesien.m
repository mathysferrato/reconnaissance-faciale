% fonction kppv.m
%
% Données :
% DataA      : les données d'apprentissage (connues)
% LabelA     : les labels des données d'apprentissage
%
% DataT      : les données de test (on veut trouver leur label)
% Nt_test    : nombre de données tests qu'on veut labelliser
% W_masque : vecteurs propres masqués
% q : nb de composantes principales
% Résultat :
% Partition : pour les Nt_test données de test, le label calculé
%
%--------------------------------------------------------------------------

function [Partition] = bayesien(DataA, LabelA, DataT, Nt_test, W_masque, q)

cx_T = (W_masque(:,1:q).')*(DataT.');
cx_A = (W_masque(:,1:q).')*(DataA.');
Partition = zeros(Nt_test,1);

for j = 1:Nt_test

    P = zeros(length(LabelA), 1);
    
    for i = 1:length(LabelA)
        [mu_i, Sigma_i] = estimation_mu_Sigma(cx_A(:,i));
        P(i) = gaussienne(cx_T(:,j).', mu_i, Sigma_i);
    end
    
    [~, indices_max] = max(P);
    
    Partition(j) = LabelA(indices_max);
end