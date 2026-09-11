# Consulta preliminar de violencia grave no Sul

Aplicacao academica end-to-end de Machine Learning supervisionado para apoiar a
consulta preliminar de municipios do Parana, Santa Catarina e Rio Grande do Sul
por empresas de construcao e planejamento urbano. O sistema produz um alerta
complementar; ele nao aprova, reprova nem substitui uma decisao de investimento.

## Proposta do trabalho

### Qual dataset foi escolhido e qual problema ele representa?

Foram escolhidas as bases anuais de Vitimas de Dados Estatisticos (VDE), de 2015
a 2025, combinadas com estimativas municipais de populacao do IBGE/SIDRA. Depois
da limpeza, a base possui 157.212 registros mensais dos 1.191 municipios da
Regiao Sul.

O problema e identificar municipios que podem apresentar uma taxa persistente
de violencia grave elevada nos seis meses seguintes. Sao considerados homicidio
doloso, latrocinio, tentativa de homicidio, lesao corporal seguida de morte e
feminicidio.

### Qual e a variavel-alvo?

O target e `alerta_taxa_violencia_grave_proximos_6m`. Ele recebe valor `1` quando:

1. a soma de ocorrencias dos seis meses seguintes supera 25 por 100 mil
   habitantes; e
2. existem ocorrencias em pelo menos dois meses diferentes desse periodo.

A segunda condicao impede que um unico caso isolado em uma cidade pequena seja
interpretado como um problema persistente.

### Quais sao as classes possiveis?

- `0 - sem alerta`: a regra proporcional e de persistencia nao foi atingida;
- `1 - com alerta`: a regra foi atingida e o municipio merece analise adicional.

### Quais informacoes entram no modelo?

O modelo usa somente informacoes conhecidas na data da consulta:

- UF e mes do ano;
- populacao municipal oficial disponivel naquele momento;
- totais recentes de violencia grave;
- valores atrasados em 1, 2, 3, 6 e 12 meses;
- medias e desvios moveis;
- taxas por 100 mil habitantes;
- taxas suavizadas para reduzir distorcoes em municipios pequenos;
- historico separado por tipo de ocorrencia.

Os seis meses futuros sao usados exclusivamente para criar o target durante o
treinamento e nunca entram como caracteristicas.

### Quem utilizaria e com qual finalidade?

Equipes de expansao, planejamento e estudos preliminares de uma construtora ou
empreiteira de cidades e bairros planejados. O alerta ajuda a priorizar locais
que precisam de uma investigacao de seguranca mais detalhada antes das etapas de
viabilidade, aquisicao de terrenos e planejamento urbano.

### O que a aplicacao fara com a classificacao?

A aplicacao apresentara a classe prevista, a probabilidade estimada e uma
orientacao operacional:

- sem alerta: continuar a consulta preliminar com os demais indicadores;
- com alerta: solicitar analise complementar de seguranca e contexto local.

O resultado nao sera apresentado como recomendacao automatica de investir ou
nao investir.

### Como sera a experiencia de uso?

O usuario selecionara UF, municipio e data de consulta em uma interface web. A
aplicacao buscara o historico necessario, enviara as variaveis para uma API
FastAPI e exibira o alerta, a probabilidade e uma explicacao curta sobre o uso
preliminar do resultado.

## Modelo selecionado

A versao proporcional usa uma `RandomForestClassifier` com 300 arvores,
profundidade maxima 14, minimo de 10 exemplos por folha e limiar 0,1313. Nas
validacoes temporais de 2020 a 2023, obteve:

- acuracia: 68,39%;
- ROC AUC: 0,7608;
- average precision: 0,3941;
- precisao: 26,53%;
- recall: 70,01%;
- F2: 0,5273.

O teste posterior ja havia sido aberto por uma versao anterior e nao foi
reutilizado para escolher este modelo. Essa limitacao esta documentada para nao
apresentar o resultado como uma avaliacao externa inedita.

## Fluxo end-to-end

```text
VDE + IBGE
    -> sanitizacao
    -> target e variaveis historicas
    -> validacao temporal
    -> treinamento e otimizacao
    -> artefato treinado
    -> API FastAPI
    -> interface de consulta
```

O projeto segue como referencia conceitual o repositorio
[`ml_fastapi_for_churn`](https://github.com/chiarorosa/ml_fastapi_for_churn).
A modelagem e o empacotamento estao concluidos; a API e a interface constituem
a proxima etapa de implementacao.

O artefato atual fica em
`models/versoes/taxa_100k/modelo_risco_taxa_100k.joblib`. Ele inclui o pipeline
de pre-processamento, a Random Forest, o limiar e os metadados necessarios para
a futura API.

## Organizacao do repositorio

```text
scripts/
|-- dados/                         # coleta, limpeza e populacao do IBGE
|-- versoes/
|   |-- maximo_movel_6m/           # versao anterior preservada
|   `-- taxa_100k/                 # versao proporcional atual
`-- experimentos/                  # tentativas e diagnosticos nao promovidos

data/processed/
|-- referencias/                   # populacao do IBGE
|-- versoes/                       # targets separados por versao
|-- experimentos/                  # previsoes usadas em diagnosticos
`-- *.csv                          # bases mensais compartilhadas

reports/
|-- dados/                         # auditoria da preparacao
|-- versoes/                       # metricas de cada versao
|-- experimentos/                  # resultados de tentativas
`-- historico/                     # trabalhos anteriores arquivados
```

Consulte [`docs/contexto_do_projeto.md`](docs/contexto_do_projeto.md) para o
historico das decisoes e
[`docs/revisao_literatura_falsos_positivos.md`](docs/revisao_literatura_falsos_positivos.md)
para a revisao utilizada na reducao de falsos alertas.

## Execucao da versao atual

No PowerShell, a partir da raiz do repositorio:

```powershell
python -m pip install -r requirements.txt
python -m scripts.dados.baixar_populacao_ibge
python -m scripts.versoes.taxa_100k.criar_target_taxa_100k
python -m scripts.versoes.taxa_100k.treinar_modelo_taxa_100k
python -m scripts.versoes.taxa_100k.otimizar_modelo_taxa_100k
python -m scripts.versoes.taxa_100k.empacotar_modelo
python -m pytest -q
```

Os experimentos antigos nao fazem parte da execucao principal. Eles permanecem
no repositorio para demonstrar as alternativas avaliadas e justificar a escolha
da versao atual.

## Limitacoes

- A precisao ainda e baixa: muitos alertas nao se confirmam.
- Ocorrencias raras em municipios pequenos continuam dificeis de prever.
- A base municipal nao permite conclusoes por bairro.
- O indicador de seguranca deve ser combinado com mercado, infraestrutura,
  mobilidade, legislacao e viabilidade financeira.
- A aplicacao e uma consulta preliminar academica, nao um sistema autonomo de
  decisao empresarial.
