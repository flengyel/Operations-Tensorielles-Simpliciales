# Opérations Tensorielles Simpliciales

**Répertoire pour l'exploration des opérations simpliciales dans les réseaux neuronaux et l'apprentissage automatique.**

Ce projet explore les principes algébriques et topologiques qui se manifestent dans les réseaux neuronaux, notamment par l'utilisation d'opérations de face, de dégénérescence et de frontière sur les matrices de poids et les hypermatrices.

## Formalités

Nous considérons les opérations simpliciales définies sur les matrices et les hypermatrices comme suit :

$$
\begin{aligned}
d_i d_j &= d_{j-1} d_i, \quad\,  \text{si } i < j; \\
s_i s_j &= s_j s_{i-1}, \quad\,  \text{si } i > j; \\
d_i s_j &=
\begin{cases}
s_{j-1} d_i,  \; \text{si } i < j; \\
1, \qquad \; \text{si } i \in \lbrace j, j+1\rbrace; \\
s_j d_{i-1},  \;\text{si } i > j+1.
\end{cases}
\end{aligned}
$$

Les opérations simpliciales satisfont aux identités simpliciales. L'un des principes fondamentaux de ce travail est l'égalité mathématique $\partial \partial = 0$, qui sert à la fois de formalisme et de contrainte philosophique sur l'introspection dans l'apprentissage automatique. Le frontière d'une frontière est nulle; cela incarne les limitations dans la capacité des mécanismes introspectifs à se comprendre eux-mêmes.

## Sous-bassements Philosophiques

L'expression $\partial \partial = 0$ capture l'essence de la frontière comme lieu de perception et de cognition. Elle nous invite à considérer que les mécanismes de la pensée et de la perception sont intrinsèquement imperceptibles à leur cœur opérationnel. Le projet explore davantage ces idées.
