# Terravista — consulta preliminar de violência grave

Aplicação end-to-end de Machine Learning supervisionado para classificação de
alerta municipal. A solução foi criada para apoiar consultas preliminares de
empresas de construção, empreiteiras e equipes de planejamento de cidades e
bairros planejados.

O usuário escolhe um município da Região Sul e um mês de referência. O sistema
recupera o histórico conhecido naquela data, executa o modelo e informa se há
indicação de alerta para os seis meses seguintes. O resultado ajuda a priorizar
uma análise complementar, mas não decide se um investimento deve ser realizado.

## Respostas exigidas pela atividade

### 1. Qual dataset foi escolhido e qual problema ele representa?

Foram utilizadas as bases anuais de Vítimas de Dados Estatísticos (VDE), entre
2015 e 2025, combinadas com estimativas municipais de população do IBGE/SIDRA.

Depois da preparação, a base mensal possui 157.212 registros referentes aos
1.191 municípios do Paraná, Santa Catarina e Rio Grande do Sul. São consideradas
cinco categorias de violência grave:

- homicídio doloso;
- tentativa de homicídio;
- latrocínio;
- feminicídio;
- lesão corporal seguida de morte.

O problema consiste em classificar se um município apresenta indicação de uma
taxa persistente de violência grave elevada nos seis meses posteriores à
consulta. A aplicação não representa toda a criminalidade municipal, pois
considera somente as categorias listadas.

### 2. Qual é a variável-alvo (target) que será prevista?

A variável-alvo é:

```text
alerta_taxa_violencia_grave_proximos_6m
```

Para cada município e mês histórico, o target recebe valor `1` quando as duas
condições abaixo são satisfeitas nos seis meses seguintes:

1. a taxa acumulada supera 25 registros por 100 mil habitantes; e
2. existem registros em pelo menos dois meses diferentes.

A taxa é calculada por:

```text
(total dos seis meses seguintes / população utilizada) × 100.000
```

O corte original, 24,877921 por 100 mil, corresponde ao percentil 80 dos dados
anteriores a 2020. Ele foi arredondado para 25 para tornar a regra compreensível.
A exigência de dois meses reduz alertas causados por um único episódio isolado
em municípios pequenos.

Os meses futuros são usados apenas para criar o gabarito durante o treinamento.
Eles nunca entram nas características fornecidas ao modelo.

Das 136.965 observações elegíveis, 18.536 receberam target positivo, o que
corresponde a 13,53% da base de modelagem.

### 3. Quais são as classes possíveis?

- `0 — Sem alerta`: a regra proporcional e de persistência não foi atingida.
- `1 — Com alerta`: a regra foi atingida e o município merece análise adicional.

O target não é uma nota de segurança, um ranking municipal ou uma recomendação
automática de investimento. Ele é uma classificação binária para triagem.

### 4. Quais informações são utilizadas como entrada do modelo?

O modelo utiliza somente dados disponíveis até o mês selecionado:

- UF e mês do ano;
- população municipal disponível naquele momento;
- total atual de violência grave;
- valores anteriores de 1, 2, 3, 6 e 12 meses;
- médias e desvios móveis;
- taxas municipais de 3, 6 e 12 meses por 100 mil habitantes;
- taxa estadual dos últimos 12 meses;
- taxas suavizadas para reduzir distorções em municípios pequenos;
- histórico de cada uma das cinco categorias;
- distância e proporção em relação ao máximo móvel observado até a consulta.

Ao todo, o pipeline recebe 44 características. O usuário não precisa preenchê-las:
a API constrói todas automaticamente a partir de UF, município e data.

### 5. Quem utilizaria a aplicação e com qual finalidade?

A aplicação foi pensada para equipes de expansão, planejamento e estudos de
viabilidade de uma construtora ou empreiteira de cidades e bairros planejados.

Ela ajuda a identificar municípios que precisam de investigação adicional de
segurança antes de etapas como análise de viabilidade, aquisição de terrenos e
planejamento urbano.

### 6. O que a aplicação faz com a classificação produzida?

A aplicação converte a probabilidade do modelo em uma orientação preliminar:

- **Sem alerta:** continuar a consulta com os demais indicadores do município.
- **Com alerta:** solicitar análise complementar de segurança e contexto local.

Além da classe, são apresentados:

- probabilidade estimada e limiar de classificação;
- população utilizada;
- registros no mês consultado;
- taxas municipais dos últimos 6 e 12 meses;
- comparação com a taxa estadual;
- tendência entre os dois períodos de 6 meses mais recentes;
- gráfico mensal dos últimos 12 meses;
- composição e categoria predominante no período.

Esses dados dão contexto à classificação, mas não explicam causalidade nem
substituem uma avaliação profissional.

### 7. Como é a interface e a experiência de uso?

A interface web apresenta um formulário com três seleções encadeadas:

1. estado;
2. município;
3. mês de referência.

Após a consulta, o usuário recebe um cartão de resultado com a classificação,
a probabilidade e os indicadores históricos. A página possui estados de
carregamento, erro com nova tentativa, layout responsivo, navegação por teclado,
tema claro/escuro e documentação interativa da API.

Consultas repetidas ao mesmo município e período são recuperadas do cache local.
O cabeçalho `X-Cache` informa `MISS` no primeiro processamento e `HIT` quando a
resposta é reaproveitada.

## Fluxo end-to-end

```text
VDE + IBGE
    ↓
Sanitização e consolidação mensal
    ↓
Criação do target e das variáveis históricas
    ↓
Comparação e validação temporal dos algoritmos
    ↓
Otimização e empacotamento do modelo
    ↓
API FastAPI
    ↓
Interface web de consulta
```

A arquitetura segue a lógica solicitada na atividade:

```text
Dataset → preparação → treinamento → avaliação → modelo → API → aplicação
```

O projeto utiliza como referência conceitual o repositório
[`ml_fastapi_for_churn`](https://github.com/chiarorosa/ml_fastapi_for_churn),
adaptando o fluxo para outro domínio de classificação.

## Modelo selecionado e avaliação

Foram comparados baseline, regressão logística, Random Forest, Extra Trees e
Gradient Boosting. KNN e árvore de decisão também foram experimentados e
preservados no histórico do projeto.

O modelo promovido foi uma `RandomForestClassifier` configurada com:

- 300 árvores;
- profundidade máxima 14;
- mínimo de 10 observações por folha;
- limiar de alerta de 0,1313;
- semente aleatória 42.

Resultados agrupados das validações temporais de 2020 a 2023:

| Métrica | Resultado |
|---|---:|
| Acurácia | 68,39% |
| ROC AUC | 0,7608 |
| Average Precision | 0,3941 |
| Precisão | 26,53% |
| Recall | 70,01% |
| F1 | 0,3848 |
| F2 | 0,5273 |

O limiar de 13,13% não é o corte do target. Os valores possuem funções diferentes:

- **25 por 100 mil:** define o gabarito real usado no treinamento;
- **13,13%:** transforma a probabilidade do modelo em uma das duas classes.

O limiar foi escolhido para manter recall próximo de 70%, pois deixar um alerta
real passar despercebido foi considerado mais prejudicial que encaminhar um
município para uma verificação adicional. Essa escolha aumenta a quantidade de
falsos positivos e está documentada como limitação.

O conjunto posterior já havia sido consultado durante uma versão anterior do
projeto e não foi reutilizado como se fosse uma avaliação externa inédita.

## Prevenção de vazamento de dados

A separação entre treino e validação respeita o tempo. Para validar determinado
ano, o treinamento utiliza apenas targets cujo horizonte de seis meses já havia
terminado antes daquele ano.

As variáveis históricas são calculadas somente com o mês da consulta e seus
meses anteriores. A população utilizada é a estimativa mais recente disponível
antes do ano consultado.

## Arquitetura da aplicação

```text
api/
├── main.py                         # inicialização e registro dos routers
├── controllers/                    # rotas e respostas HTTP
├── services/                       # regras de negócio e execução do modelo
├── schemas.py                      # contratos de entrada e saída
├── cache.py                        # cache LRU com expiração
└── static.py                       # entrega dos arquivos da interface

frontend/
├── index.html                      # estrutura semântica da interface
├── styles.css                      # design responsivo e temas
└── app.js                          # integração com a API

scripts/
├── dados/                          # coleta e preparação
├── versoes/
│   ├── maximo_movel_6m/            # versão anterior preservada
│   └── taxa_100k/                  # versão atual
└── experimentos/                   # comparações não promovidas

models/versoes/taxa_100k/           # modelo empacotado e metadados
data/processed/                     # bases processadas
reports/                            # métricas e auditorias
tests/                              # testes automatizados
```

O artefato `modelo_risco_taxa_100k.joblib` contém o pré-processamento, a Random
Forest treinada, o limiar escolhido e os metadados necessários para inferência.

## Endpoints da API

| Método | Endpoint | Finalidade |
|---|---|---|
| `GET` | `/api/v1/health` | Verificar API, modelo e cache |
| `GET` | `/api/v1/metadata` | Consultar metadados do modelo |
| `GET` | `/api/v1/municipios?uf=PR` | Listar municípios disponíveis |
| `GET` | `/api/v1/periodos` | Listar períodos de um município |
| `POST` | `/api/v1/predicoes` | Produzir classificação e contexto |

Exemplo de entrada:

```json
{
  "uf": "PR",
  "municipio": "Curitiba",
  "data_referencia": "2025-06-01"
}
```

## Como executar localmente

Requisitos: Python e `pip` disponíveis no terminal.

```powershell
python -m pip install -r requirements.txt
python -m uvicorn api.main:app --reload
```

Depois da inicialização:

- interface: [http://127.0.0.1:8000](http://127.0.0.1:8000);
- documentação Swagger: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs);
- documentação ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc).

A primeira inicialização pode levar alguns segundos porque o modelo e o histórico
municipal são carregados em memória.

### Executar os testes

```powershell
python -m pytest -q
```

### Reproduzir a versão atual do modelo

```powershell
python -m scripts.dados.baixar_populacao_ibge
python -m scripts.versoes.taxa_100k.criar_target_taxa_100k
python -m scripts.versoes.taxa_100k.treinar_modelo_taxa_100k
python -m scripts.versoes.taxa_100k.otimizar_modelo_taxa_100k
python -m scripts.versoes.taxa_100k.empacotar_modelo
```

Os experimentos antigos não fazem parte da execução principal. Eles foram
mantidos para documentar as alternativas avaliadas e justificar a versão atual.

## Cache

A API mantém até 4.096 predições em um cache LRU local com validade de uma hora.
O cache utiliza UF, município e período como chave. Ele é suficiente para a
execução acadêmica em uma única instância e pode ser substituído por Redis em
uma implantação distribuída.

O cache das predições não altera o modelo nem as probabilidades: ele apenas
reaproveita uma resposta que já foi calculada para a mesma consulta.

## Limitações e uso responsável

- A precisão de 26,53% significa que parte relevante dos alertas não se confirma.
- O modelo identifica padrões históricos, mas não determina as causas da violência.
- Ocorrências raras em municípios pequenos continuam difíceis de prever.
- A base é municipal e não permite conclusões por bairro.
- As cinco categorias não representam toda a criminalidade local.
- Mudanças de registro, subnotificação e qualidade das fontes afetam o resultado.
- Segurança deve ser analisada junto de mercado, infraestrutura, mobilidade,
  legislação, viabilidade financeira e conhecimento local.
- A aplicação é uma consulta preliminar acadêmica, não um sistema autônomo de
  decisão empresarial ou de segurança pública.

## Documentação complementar

- [`docs/contexto_do_projeto.md`](docs/contexto_do_projeto.md): evolução das
  decisões, experimentos e cuidados metodológicos.
- [`docs/revisao_literatura_falsos_positivos.md`](docs/revisao_literatura_falsos_positivos.md):
  referências utilizadas na investigação dos falsos positivos.
- [`reports/versoes/taxa_100k/`](reports/versoes/taxa_100k/): relatórios da
  versão promovida.
