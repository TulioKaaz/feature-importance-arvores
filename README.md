# Feature Importance em Árvores — Prática

Prática do Bloco 4 da trilha de seleção de features (StatQuest + *Advances in Financial Machine Learning*, cap. 8): comparar os métodos de feature importance em modelos de árvore — MDI, permutation importance (treino vs. teste), os três rankings do XGBoost (`weight`, `gain`, `cover`) e SHAP (TreeSHAP) — sobre um dataset simulado com armadilhas plantadas de propósito.

Material de estudo: nota "Feature Importance (Árvores)", no vault pessoal (Obsidian), dentro de *Machine Learning - Seleção de Features*.

## Roteiro

1. Simular um dataset de regressão com features legítimas e três armadilhas: ruído puro, categórica de alta cardinalidade aleatória e uma feature duplicada de uma das fortes.
2. Treinar um modelo de árvore (XGBoost) sobre todas as features.
3. Comparar MDI (`feature_importances_`), permutation importance (treino vs. teste), os três rankings nativos do XGBoost (`weight`, `gain`, `cover`) e o ranking por média de |SHAP| (TreeSHAP).
4. Analisar: a feature de ruído aparece com importância não-nula em algum método? O que acontece com o ranking da feature duplicada vs. a original? Os três tipos do XGBoost concordam entre si? O viés de treino aparece no permutation importance? SHAP corrige alguma inconsistência mostrada pelos métodos anteriores?

## Estrutura

- `notebooks/feature_importance_arvores.ipynb` — notebook principal.
- `requirements.txt` — dependências Python do projeto.
- `outputs/` — gráficos e tabelas gerados pelo notebook.

## Como rodar

> Convenção de commits: [Conventional Commits](https://www.conventionalcommits.org/) (`feat`, `fix`, `docs`, `refactor`, `chore`, etc.), uma etapa do notebook por commit e só depois de rodar sem erro — mantém o histórico legível.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## Licença

GPLv3 — ver [LICENSE](LICENSE).
