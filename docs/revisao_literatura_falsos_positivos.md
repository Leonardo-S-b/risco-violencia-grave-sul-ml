# Revisao de literatura para reduzir falsos positivos

## Pergunta do projeto

Como reduzir falsos alertas ao prever, por municipio, se a violencia grave dos
proximos seis meses superara o maximo conhecido?

O problema observado no projeto se concentra nos municipios de baixo volume. Em
uma cidade com muitos meses zerados, uma unica ocorrencia pode criar um novo
maximo. O rotulo passa a representar em parte uma oscilacao rara e aleatoria, e
nao apenas uma mudanca persistente de risco.

Restricao do trabalho: todas as propostas devem preservar a classificacao
binaria (`0 = sem alerta`, `1 = com alerta`). Taxas, suavizacao bayesiana e
informacoes espaciais serao usadas como variaveis ou na definicao do target, sem
trocar a tarefa final por regressao.

## Conclusao principal

A evidencia mais diretamente aplicavel recomenda trocar a comparacao de contagens
brutas por uma medida ajustada pela populacao e suavizada para municipios
pequenos. Depois, devem ser adicionadas variaveis espaciais dos municipios
vizinhos. Calibracao e classificacao seletiva podem melhorar a forma como o
alerta e emitido, mas nao substituem uma representacao melhor do risco.

## 1. Taxa populacional com suavizacao bayesiana

Carvalho et al. estudam taxas de homicidio nos municipios brasileiros e mostram
que eventos raros e populacoes pequenas tornam as taxas brutas muito variaveis.
Uma unica ocorrencia pode fazer um municipio pequeno parecer extremamente
perigoso por acaso. Os autores recomendam estimadores bayesianos, especialmente
com componente espacial, para aproximar estimativas instaveis da media global ou
local com intensidade inversamente relacionada ao tamanho da populacao.

Aplicacao proposta:

1. Acrescentar populacao anual de cada municipio.
2. Calcular taxa mensal ou semestral por 100 mil habitantes.
3. Suavizar a taxa dos municipios pequenos em direcao a media da UF ou dos
   vizinhos.
4. Comparar o target atual com um target baseado na taxa suavizada.

Esta e a hipotese com maior chance de atacar a origem dos falsos positivos dos
perfis 1 e 2.

Fonte: Carvalho AX, Albuquerque PHM, Silva GDM, Almeida Junior GR. "Taxas
bayesianas para o mapeamento de homicidios nos municipios brasileiros".
https://www.scielo.br/j/csp/a/zw9hkPGwyWkDKV34FsDyKFL/?lang=pt

## 2. Padronizacao demografica e modelos de contagem

Wanzinack, Signorelli e Reis calculam taxas padronizadas municipais, comparam o
numero observado ao esperado e usam regressao de Poisson com efeito
espaco-temporal. O estudo reforca que a unidade correta e a populacao sob risco,
nao apenas a contagem absoluta, e que municipios de um mesmo estado sao
heterogeneos.

Aplicacao proposta:

- usar populacao como exposicao;
- testar a razao observado/esperado como variavel;
- usar os conceitos de valor observado e esperado como novas variaveis do
  classificador;
- manter Gradient Boosting, regressao logistica classificatoria e demais
  classificadores como candidatos.

Os modelos de Poisson do artigo ficam apenas como referencia estatistica e nao
serao adotados, pois mudariam o foco exigido pelo trabalho.

Fonte: Wanzinack C, Signorelli MC, Reis C. "Mortalidade por homicidios nas
unidades da federacao e nos municipios brasileiros de 2005 a 2015: uma analise
socioespacial".
https://www.scielo.br/j/cadsc/a/q5HWzZ9XQqNWv4jBZppqQGQ/?lang=pt

## 3. Dependencia espacial e defasagens dos vizinhos

Um estudo de previsao de risco criminal em Dallas relata melhora ao acrescentar
defasagens espaco-temporais. A ideia e que o historico do proprio municipio nao
contem toda a informacao: crescimento ou deslocamento da violencia em municipios
proximos pode anteceder mudancas locais.

Aplicacao proposta:

- obter codigo IBGE e geometria ou lista de municipios limitrofes;
- criar media, maximo e tendencia dos vizinhos nos ultimos 3, 6 e 12 meses;
- calcular Moran local ou uma versao simples da defasagem espacial;
- manter a separacao temporal durante a construcao das variaveis.

Fonte: "Crime risk prediction incorporating geographical spatiotemporal
dependency into machine learning models", Information Sciences 646 (2023),
119414. https://doi.org/10.1016/j.ins.2023.119414

## 4. Tratamento especifico para regioes pouco populosas

Kadar et al. tratam previsao criminal em areas de baixa densidade como um
problema proprio, marcado por forte esparsidade e desequilibrio. O trabalho usa
variaveis criminais, temporais, socioeconomicas, geograficas e meteorologicas e
avalia a recuperacao de crimes dentro dos principais hotspots, em vez de exigir
uma decisao positiva ou negativa para toda localidade.

Aplicacao proposta:

- avaliar `precision@k` e `recall@k`;
- apresentar somente os 5%, 10% ou 20% maiores riscos para investigacao;
- comparar o desempenho separadamente por porte populacional;
- evitar que milhares de previsoes de baixa confianca virem alertas.

Fonte: "Public decision support for low population density areas: An
imbalance-aware hyper-ensemble for spatio-temporal crime prediction", Decision
Support Systems 130 (2020), 113242.
https://doi.org/10.1016/j.dss.2019.113242

## 5. Classificacao seletiva como referencia futura

Gangrade, Kag e Saligrama estudam classificadores que podem se abster quando nao
ha confianca suficiente. A abordagem troca cobertura por menos erros e procura
explicitamente conjuntos de decisao com poucos falsos positivos.

Aplicacao proposta ao sistema:

- probabilidade alta: `alerta`;
- probabilidade baixa: `sem alerta`;
- faixa intermediaria: `inconclusivo; requer analise complementar`.

Isso combina com a finalidade de consulta preliminar, mas transformaria a saida
binaria em tres respostas. Portanto, fica apenas como possibilidade futura e nao
sera usada enquanto a exigencia for classificacao binaria estrita.

Fonte: Gangrade A, Kag A, Saligrama V. "Selective Classification via One-Sided
Prediction", AISTATS 2021.
https://proceedings.mlr.press/v130/gangrade21a.html

## 6. Calibracao das probabilidades

Kull, Silva Filho e Flach mostram que decisoes com custos diferentes exigem
probabilidades calibradas e apresentam a calibracao beta. No projeto, o valor
0,1081 e hoje apenas um escore usado como limite; ainda nao foi demonstrado que
10,81% de previsao corresponde a aproximadamente 10,81% de frequencia real.

Aplicacao proposta:

- gerar previsoes fora do treino nas validacoes temporais;
- comparar sigmoid/Platt, isotonic e beta calibration;
- medir Brier score, curva de confiabilidade e erro de calibracao;
- somente depois definir limites de custo ou faixas inconclusivas.

Calibracao torna a probabilidade interpretavel, mas nao melhora necessariamente
o ranking AP nem reduz falsos positivos mantendo o mesmo recall.

Fonte: Kull M, Silva Filho T, Flach P. "Beta calibration: a well-founded and
easily implemented improvement on logistic calibration for binary classifiers",
AISTATS 2017. https://proceedings.mlr.press/v54/kull17a.html

## 7. Indicadores antecedentes e interacoes espaciais

Cohen, Gorr e Olligschlaeger usam crimes antecedentes, defasagens temporais e
medias de regioes contiguas para prever mudancas de volume. No estudo, os modelos
encontraram parte relevante das grandes mudancas com uma carga de investigacao
controlada e precisao relatada de 31% nos casos investigados.

Aplicacao proposta:

- manter separadas as cinco categorias de violencia grave;
- testar se tentativas de homicidio antecedem homicidios e outros picos;
- incluir mudancas nos municipios vizinhos como indicadores antecedentes;
- avaliar quantidade mensal de casos encaminhados para consulta humana.

Fonte: Cohen J, Gorr WL, Olligschlaeger AM. "Leading Indicators and Spatial
Interactions: A Crime-Forecasting Model for Proactive Police Deployment",
Geographical Analysis 39(1), 2007.
https://doi.org/10.1111/j.1538-4632.2006.00697.x

## Ordem recomendada de experimentos

1. Integrar populacao municipal anual e criar taxas por 100 mil habitantes.
2. Aplicar suavizacao bayesiana por UF e, quando houver adjacencia, por vizinhos.
3. Comparar target bruto, target suavizado e aumento sustentado em pelo menos dois
   meses do horizonte.
4. Acrescentar defasagens dos municipios vizinhos.
5. Calibrar probabilidades usando previsoes fora do treino.
6. Manter a saida binaria e comparar limites definidos apenas na validacao.
7. Reportar precision/recall por perfil e `precision@k` para um orcamento fixo de
   consultas.

## O que nao priorizar agora

- Mais undersampling: os experimentos do projeto ja mostraram que nao resolveu.
- Redes neurais ou GNN imediatamente: exigiriam uma estrutura espacial pronta,
  maior complexidade e justificativa que a base atual ainda nao oferece.
- Ajustar novamente o limite no teste final: o periodo ja foi aberto e nao deve
  orientar novas escolhas sem ser declarado como analise pos-teste.
