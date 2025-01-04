Ce premier graphique illustre la comparaison des prix d'options calculés par différentes méthodes en fonction de la dimension.

Points clés à souligner :

- La méthode BSDE-DNN maintient une stabilité relative même en haute dimension
- La méthode Longstaff-Schwartz montre une déviation croissante avec la dimension
- Les différences de prix deviennent plus prononcées au-delà de 10 dimensions
- La méthode des différences finies n'est applicable qu'en basse dimension

Cette comparaison démontre l'avantage principal de l'approche par réseaux de neurones : sa capacité à gérer efficacement les problèmes en haute dimension."

Le deuxième graphique présente l'analyse des temps de calcul en échelle logarithmique.

Points essentiels :

- L'axe vertical en échelle logarithmique montre la croissance exponentielle du temps de calcul
- La méthode BSDE-DNN montre une meilleure scalabilité en haute dimension
- Les méthodes traditionnelles (Longstaff-Schwartz, différences finies) deviennent prohibitives au-delà de certaines dimensions
- Le compromis temps/précision est particulièrement favorable pour BSDE-DNN en dimensions élevées

Cette analyse démontre clairement l'efficacité computationnelle de notre approche par rapport aux méthodes classiques."

Le troisième graphique présente l'analyse de l'erreur relative en pourcentage.

Points à mettre en avant :

- L'erreur est calculée par rapport à une solution de référence (Black-Scholes en 1D)
- La stabilité de l'erreur pour BSDE-DNN même en haute dimension
- L'augmentation progressive de l'erreur pour Longstaff-Schwartz
- La limitation des méthodes traditionnelles visible par leur erreur croissante

Cette analyse confirme la robustesse de notre approche BSDE-DNN, particulièrement importante pour les applications pratiques."

En synthèse, ces résultats démontrent trois avantages majeurs de notre approche BSDE-DNN :

1. Une précision stable même en haute dimension
2. Un temps de calcul plus favorable que les méthodes traditionnelles
3. Une robustesse accrue face à l'augmentation de la dimensionnalité

Ces caractéristiques sont particulièrement pertinentes pour les applications réelles en finance, où la rapidité et la précision sont cruciales.
