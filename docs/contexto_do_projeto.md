# Contexto do projeto

## Finalidade

Este e um trabalho academico de machine learning para uma empresa ficticia de
construcao de cidades e bairros planejados. A ferramenta funciona como consulta
preliminar de seguranca: ela sinaliza municipios que merecem uma analise mais
detalhada, mas nao toma nem recomenda decisoes de investimento.

A base atual cobre os municipios do Parana, de Santa Catarina e do Rio Grande do
Sul. Como os dados sao municipais, o projeto nao permite conclusoes por bairro.

## Problema de classificacao atual

Restricao obrigatoria do trabalho: o problema deve permanecer como classificacao
binaria. A saida final e sempre `0 = sem alerta` ou `1 = com alerta`. Metodos de
regressao de contagem encontrados na literatura podem servir como referencia
teorica, mas nao devem substituir o classificador do projeto.

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

Essa hipotese foi refinada em um segundo experimento: cada municipio recebeu um
perfil fixo durante o ano, calculado somente com os 24 meses anteriores. No
criterio F2, essa versao elevou a AP para 0,356734, reduziu os falsos positivos de
28.516 para 28.293 e os falsos negativos de 1.496 para 1.371. Limites especificos
por perfil reduziram os falsos positivos para 28.283 e os falsos negativos para
1.331, ainda uma melhora pequena.

Os perfis 1 e 2, formados principalmente por municipios de volume muito baixo,
concentraram 20.580 dos 28.283 falsos positivos. Nesses municipios, uma passagem
de zero para uma ocorrencia pode constituir um novo maximo e ser essencialmente
aleatoria. Para reduzir muito os falsos alertas, a proxima decisao provavelmente
tera de envolver a regra do target para municipios de baixo volume, dados
adicionais como populacao ou uma politica operacional que aceite recall menor.
Nao e indicado criar um modelo separado por municipio, pois cada cidade possui
poucos meses e poucos casos positivos.

## Populacao e comparacao dos quatro metodos

Foram baixadas estimativas oficiais da populacao municipal do IBGE SIDRA, tabela
6579, variavel 9324, para 2013 a 2021. A integracao cobriu os 1.191 municipios da
Regiao Sul. Cada consulta usa a ultima estimativa publicada antes do seu ano; os
anos posteriores a 2021 carregam o ultimo valor conhecido.

O target permaneceu binario e inalterado. Foram acrescentados log da populacao,
taxas por 100 mil habitantes, taxas acumuladas de 3, 6 e 12 meses, taxa da UF e
tres versoes suavizadas para cidades pequenas.

Na comparacao com recall minimo de 70%, os resultados foram:

1. populacao e taxas suavizadas: AP 0,390960, precisao 0,312871, 17.126 FP;
2. perfis mensais: AP 0,358950, precisao 0,301786, 18.060 FP;
3. perfis anuais: AP 0,356734, precisao 0,300898, 18.134 FP;
4. modelo original: AP 0,331073, precisao 0,295915, 18.716 FP.

O metodo mais benefico foi populacao com taxas suavizadas. Essa decisao usou
somente as validacoes de 2020 a 2023, nunca o teste final ja aberto.

## Candidata populacional - versao 2

Depois da escolha das variaveis populacionais, foram comparadas 13 configuracoes
do HistGradientBoosting. O criterio foi minimizar falsos positivos mantendo
recall agrupado de pelo menos 70%. A configuracao `folhas_100` venceu com:

- `learning_rate=0.08`;
- `max_iter=150`;
- `max_leaf_nodes=31`;
- `min_samples_leaf=100`;
- `l2_regularization=0.1`;
- limiar unico de 0,182510.

Nas validacoes agrupadas, ela obteve AP 0,384574, precisao 0,313681, recall
0,700180, F2 0,561749, 17.066 falsos positivos e 3.340 falsos negativos. Isso
representa apenas 60 falsos positivos a menos que a configuracao populacional
anterior; portanto, a inclusao da populacao foi mais importante do que o ajuste
fino dos parametros.

Com o mesmo limiar em cada ano, o recall foi 0,623012 em 2020, 0,769651 em 2021,
0,740058 em 2022 e 0,672782 em 2023. Assim, a meta de 70% vale para o conjunto
agrupado e nao e garantida em todo ano isoladamente. Essa variacao deve ser
declarada como limitacao da candidata. O teste final de 2024-07 a 2025-06 nao
foi reutilizado nesta etapa.

## Diagnostico dos falsos positivos

Nas 57.168 previsoes fora do treino, 8.526 dos 17.066 falsos positivos ocorreram
quando o maximo historico dos 12 meses era zero. Nesse contexto, qualquer unica
ocorrencia futura gera target positivo, mas as cidades apresentam pouco sinal
criminal recente que permita antecipar esse evento raro. Foram encontrados
municipios com alerta falso em todos os 48 meses de validacao.

Dois limiares contextuais, um para maximo zero e outro para maximo positivo,
reduziram os falsos positivos de 17.066 para 16.954 com recall agrupado de 70%.
O ganho de 112 erros foi pequeno e nao justifica promover a politica sem nova
validacao.

Tambem foram testadas medias, desvios, taxas e frequencia de ocorrencias em 24 e
36 meses, alem do tempo desde a ultima ocorrencia. O resultado foi de 17.000
falsos positivos, apenas 66 a menos que a versao 2. Houve melhora em 2020, mas
piora em 2022 e 2023. Essas variaveis permanecem como experimento, nao como nova
versao escolhida.

A conclusao atual e que aproximadamente metade dos falsos alertas esta ligada a
municipios sem ocorrencia nos 12 meses anteriores. Ajustes de limiar e extensao
do historico ajudam pouco; uma reducao grande provavelmente exige fontes
explicativas adicionais ou uma revisao explicitamente justificada da regra de
alerta para historico zero.

## Experimento de target proporcional

Dividir o valor futuro e o maximo historico pela mesma populacao nao altera a
classificacao: as 136.965 linhas elegiveis conservaram exatamente o mesmo
rotulo. Para a proporcionalidade ter efeito real, foi criada outra pergunta de
negocio: se a taxa acumulada nos proximos seis meses ultrapassara 24,877921
ocorrencias por 100 mil habitantes. Esse corte corresponde ao percentil 80 do
periodo anterior a 2020.

Como taxas brutas tambem podem exagerar um unico evento em cidades pequenas,
foram comparadas tres definicoes: taxa pura, taxa com pelo menos dois casos e
taxa com ocorrencias em pelo menos dois meses diferentes. A ultima representa
melhor a ideia de nao tratar um caso isolado como problema persistente. Ela teve
prevalencia de 13,8% antes de 2020 e, nas validacoes, AP 0,386828, precisao
0,266166, recall 0,700136, 15.649 falsos positivos e 2.431 falsos negativos.

Esses numeros nao devem ser comparados diretamente aos erros do target anterior,
pois a resposta supervisionada e o significado do alerta mudaram. O novo target
permanece candidato e precisa de otimizacao e validacao proprias antes de
substituir a versao 2.

## Novo target proporcional escolhido para desenvolvimento

A regra foi consolidada com um corte arredondado e explicavel: target 1 quando
a taxa acumulada de violencia grave nos proximos seis meses supera 25 por 100
mil habitantes e existem ocorrencias em pelo menos dois meses diferentes. Um
unico episodio em uma cidade pequena, portanto, nao gera rotulo positivo.

O novo arquivo possui 136.965 linhas elegiveis e 18.536 positivas, equivalentes
a 13,5334%. A populacao e sempre a ultima estimativa do IBGE disponivel antes do
ano da consulta. O target anterior e seus modelos foram preservados como
historico.

Os algoritmos foram comparados novamente nas validacoes temporais de 2020 a
2023. Random Forest e Gradient Boosting foram os finalistas. Depois de uma grade
curta, venceu a Random Forest com 300 arvores, profundidade maxima 14, minimo de
10 exemplos por folha e limiar unico 0,1313. Nas validacoes agrupadas, obteve AP
0,394136, precisao 0,265318, recall 0,700074, F2 0,527273, 15.648 falsos
positivos e 2.421 falsos negativos.

Com o mesmo limiar, o recall anual foi 0,691547 em 2020, 0,717102 em 2021,
0,705128 em 2022 e 0,686245 em 2023. A variacao e menor que na candidata
anterior, embora a meta de 70% ainda nao seja atingida em todos os anos. O teste
final ja aberto nao foi reutilizado.

## Arquivos principais

- `scripts/dados/`: sanitizacao e coleta da populacao municipal.
- `scripts/versoes/maximo_movel_6m/`: pipeline completo da versao anterior,
  incluindo target, treinamento, balanceamento, otimizacao e avaliacao.
- `scripts/versoes/taxa_100k/`: target, treinamento e otimizacao da versao
  proporcional atual.
- `scripts/experimentos/`: perfis municipais, limiares, historico longo,
  diagnosticos e comparacoes que nao foram promovidos.
- `tests/test_pipeline_modelagem.py`: testes automatizados do pipeline.
- `docs/revisao_literatura_falsos_positivos.md`: artigos e roteiro de melhorias
  para reduzir falsos alertas.

## Registros historicos

Targets anteriores baseados no maximo absoluto e no maximo movel mensal foram
mantidos em `reports/historico/maximo_absoluto/` e
`reports/historico/maximo_movel_12m_mensal/`. Eles nao devem substituir o
target atual, mas servem para documentar a evolucao do trabalho.
