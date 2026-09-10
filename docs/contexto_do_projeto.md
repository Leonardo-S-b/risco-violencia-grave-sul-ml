# Contexto do projeto

## Finalidade

Este e um trabalho academico de machine learning para uma empresa ficticia de
construcao de cidades e bairros planejados. A ferramenta funciona como consulta
preliminar de seguranca: ela sinaliza municipios que merecem uma analise mais
detalhada, mas nao toma nem recomenda decisoes de investimento.

A base atual cobre os municipios do Parana, de Santa Catarina e do Rio Grande do
Sul. Como os dados sao municipais, o projeto nao permite conclusoes por bairro.

## Problema de classificacao atual

Para cada municipio e mes de consulta `t`, o target vale `1` quando pelo menos um
dos seis meses seguintes (`t+1` a `t+6`) supera o maior total mensal de violencia
grave observado nos 12 meses conhecidos ate `t`.

Coluna do target:

`alerta_violencia_grave_proximos_6m`

Distribuicao da base elegivel:

- 136.965 observacoes;
- 25.856 casos positivos;
- 18,8778% de positivos.

Os eventos agregados sao homicidio doloso, latrocinio, tentativa de homicidio,
lesao corporal seguida de morte e feminicidio.

## Cuidados contra vazamento temporal

- As variaveis usam somente informacoes conhecidas na data da consulta.
- Cada target possui `data_fim_horizonte`.
- Um exemplo so entra no treino quando seu horizonte de seis meses terminou
  antes do inicio da validacao ou do teste.
- As validacoes temporais foram feitas em 2020, 2021, 2022 e 2023.
- O teste final ficou reservado para consultas de 2024-07 a 2025-06.

## Experimentos realizados

Foram comparados baseline, regressao logistica, Random Forest, Extra Trees e
HistGradientBoosting. Decision Tree e KNN foram preservados comentados no codigo.
Tambem foram testados pesos de classe e undersampling 1:3 e 1:2. O melhor
resultado de validacao veio dos dados originais, sem balanceamento artificial.

O modelo escolhido foi o HistGradientBoosting `conservador_15`:

- `learning_rate=0.05`;
- `max_iter=250`;
- `max_leaf_nodes=15`;
- `min_samples_leaf=50`;
- `l2_regularization=1.0`;
- `early_stopping=False`;
- sem balanceamento;
- limiar definido na validacao: 0,1081.

Validacao agrupada de 2020 a 2023:

- AP: 0,331073;
- precisao: 0,252725;
- recall: 0,865709;
- F2: 0,582930.

## Teste final ja aberto

O teste final foi acessado uma unica vez com a configuracao e o limiar congelados.
Resultados nas 14.292 observacoes:

- acuracia: 0,442695;
- ROC AUC: 0,690932;
- AP: 0,274118;
- precisao: 0,208878;
- recall: 0,877163;
- F1: 0,337410;
- F2: 0,534895;
- verdadeiros negativos: 4.299;
- falsos positivos: 7.681;
- falsos negativos: 284;
- verdadeiros positivos: 2.028.

O resultado encontra aproximadamente 88 de cada 100 casos reais, mas gera
muitos falsos alertas. Ele deve permanecer registrado como baseline final. Como
o teste ja foi visto, novas escolhas nao podem ser apresentadas como se viessem
de um teste final independente.

## Analise de limiares

Somente aumentar o limiar nao resolveu o excesso de falsos positivos. Nas
validacoes, o melhor limiar por F1 foi aproximadamente 0,206: a precisao subiu
para 0,301149, mas o recall caiu para 0,684560. Ainda restaram mais de dois
falsos positivos para cada verdadeiro positivo.

## Perfis municipais

Foi levantada a hipotese de que municipios muito diferentes estavam sendo
avaliados pela mesma regua. Um primeiro experimento agrupou linhas mensais em
cinco perfis usando volume, volatilidade, proporcao de meses zerados, amplitude e
tendencia. Esse desenho melhorou a AP de 0,331073 para 0,358950, mas, no limiar
escolhido pelo F2, aumentou os falsos positivos de 28.516 para 29.511 nas
validacoes agrupadas. Portanto, essa versao nao foi promovida a modelo final.

A proxima hipotese e agrupar municipios pelo comportamento historico consolidado,
em vez de permitir que cada linha mensal receba um perfil diferente. Depois,
devem ser avaliados limites de alerta por perfil, sempre apenas nas validacoes
temporais. Nao e indicado criar um modelo separado por municipio, pois cada
cidade possui poucos meses e poucos casos positivos.

## Arquivos principais

- `scripts/criar_target_violencia_grave.py`: criacao do target.
- `scripts/treinar_modelo_risco.py`: variaveis e comparacao inicial.
- `scripts/comparar_balanceamento.py`: estrategias de balanceamento.
- `scripts/otimizar_gradient_boosting.py`: busca de hiperparametros.
- `scripts/validar_estabilidade_modelo.py`: limiar unico e estabilidade anual.
- `scripts/analisar_limiares_modelo.py`: troca entre precisao e recall.
- `scripts/comparar_perfis_municipios.py`: primeiro experimento com perfis.
- `scripts/avaliar_modelo_final.py`: avaliacao final e artefato do modelo.
- `tests/test_pipeline_modelagem.py`: testes automatizados do pipeline.

## Registros historicos

Targets anteriores baseados no maximo absoluto e no maximo movel mensal foram
mantidos em `reports/experimento_maximo_absoluto/` e
`reports/experimento_maximo_movel_12m_mensal/`. Eles nao devem substituir o
target atual, mas servem para documentar a evolucao do trabalho.
