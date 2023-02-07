
% fonction kppv.m
%
% Données :
% DataA      : les données d'apprentissage (connues)
% LabelA     : les labels des données d'apprentissage
%
% DataT      : les données de test (on veut trouver leur label)
% Nt_test    : nombre de données tests qu'on veut labelliser
%
% K          : le K de l'algorithme des k-plus-proches-voisins
% ListeClass : les classes possibles (== les labels possibles)
% W_masque : vecteurs propres masqués
% q : nb de composantes principales
% Résultat :
% Partition : pour les Nt_test données de test, le label calculé
%
%--------------------------------------------------------------------------
function [Partition] = kppv(DataA, LabelA, DataT, Nt_test, K, ListeClass, W_masque, q)

cx_T = (W_masque(:,1:q).')*(DataT.');
cx_A = (W_masque(:,1:q).')*(DataA.');
[~,Na] = size(cx_A);

% Initialisation du vecteur d'étiquetage des images tests
Partition = zeros(Nt_test,1);

% Boucle sur les vecteurs test de l'ensemble de l'évaluation
for i = 1:Nt_test

    % Calcul des distances entre le vecteur de test 
    % et les vecteurs d'apprentissage (voisins)
    d_x_xi = sum((cx_T(:,i)*ones(1,Na) - cx_A).^2,1);
    
    % On ne garde que les indices des K + proches voisins
    [~, indices] = sort(d_x_xi, 'ascend');
    indices_kppv = indices(1:K);
    
    % Comptage du nombre de voisins appartenant à chaque classe
    nb_classes = length(ListeClass);
    nb_occ = zeros(1,nb_classes);
    for k = 1:nb_classes
        nb_occ(k) = length(find(LabelA(indices_kppv) == ListeClass(k)));
    end
    
    % Recherche des classes contenant le maximum de voisins
    max_occ = max(nb_occ);
    indices_max = find(nb_occ == max_occ);
    
    % Si l'image test a le plus grand nombre de voisins dans plusieurs  
    % classes différentes, alors on lui assigne celle du voisin le + proche,
    % sinon on lui assigne l'unique classe contenant le plus de voisins 
    if (length(indices_max) == 1)
        p = ListeClass(indices_max);
    else
        p = LabelA(indices_kppv(1));
    end

    % Assignation de l'étiquette correspondant à la classe trouvée au point 
    % correspondant à la i-ème image test dans le vecteur "Partition" 
    Partition(i) = p;

end

