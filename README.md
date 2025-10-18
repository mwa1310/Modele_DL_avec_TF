Ce projet s’inscrit dans le cadre du module d’Apprentissage Profond et Ingénierie des Systèmes Intelligents.
L’objectif est de guider l’étudiant à travers tout le cycle de vie d’un modèle de Deep Learning, depuis sa conception jusqu’à son déploiement en environnement de production.

Dans la première partie, nous construisons un réseau de neurones dense (fully connected) avec Keras pour la classification d’images du jeu de données MNIST (chiffres manuscrits).
Le modèle est entraîné, évalué puis sauvegardé, tout en explorant les notions fondamentales du deep learning : descente de gradient, rétropropagation, fonctions d’activation et régularisation.

La deuxième partie aborde la dimension ingénierie du Deep Learning :

  - Versionnement du code avec Git et GitHub pour assurer la traçabilité et la collaboration.

  - Suivi des expérimentations avec MLflow, permettant d’enregistrer les paramètres, métriques et modèles.

  - Création d’une API Flask pour exposer le modèle en tant que service prédictif.

  - Conteneurisation avec Docker afin d’assurer la portabilité du modèle.

  - Déploiement automatisé via un pipeline CI/CD (GitHub Actions → Cloud Run / Amazon ECS) et mise en place d’un monitoring post-déploiement.

Ce TP illustre donc le passage de la recherche à la production dans un projet de Deep Learning moderne, en appliquant à la fois les bonnes pratiques de développement logiciel et de gestion de modèles IA.
