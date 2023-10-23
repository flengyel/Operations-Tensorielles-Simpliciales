# Opérations Tensorielles Simpliciales

**Répertoire pour l'exploration des opérations simpliciales sur les matrices et les hypermatrices.**

## Formalités Mathématiques

Ce projet se concentre sur les opérations simpliciales sur les matrices et hypermatrices, en particulier les opérations de face (`d_i`), de dégénérescence (`s_i`), et de bord. Les opérations étudiées ici satisfont aux identités simpliciales.

$$
\begin{aligned}
d_i d_j &= d_{j-1} d_i, \quad  \text{ si } i < j; \\
s_i s_j &= s_j s_{i-1}, \quad  \text{ si } i > j; \\
d_i s_j &=
\begin{cases}
s_{j-1} d_i,   \text{ si } i < j; \\
1, \qquad  \text{ si } i \in \lbrace j, j+1\rbrace; \\
s_j d_{i-1},  \text{ si } i > j+1.
\end{cases}
\end{aligned}
$$
