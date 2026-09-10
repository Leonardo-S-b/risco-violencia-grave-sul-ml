# Consulta preliminar de violencia grave no Sul

Projeto academico de machine learning para apoiar uma consulta preliminar sobre
violencia grave nos municipios do Parana, de Santa Catarina e do Rio Grande do
Sul. O resultado e um alerta complementar para estudos iniciais de empreendimentos
urbanos; nao constitui recomendacao ou decisao de investimento.

## Objetivo de modelagem

Para cada municipio e data de consulta, o alvo vale `1` quando pelo menos um dos
seis meses seguintes supera o maior total mensal conhecido nos 12 meses ate a
consulta. O modelo continua sendo de classificacao e usa somente informacoes
disponiveis na data consultada. Os experimentos mensais anteriores permanecem
arquivados em `reports/experimento_maximo_absoluto/` e
`reports/experimento_maximo_movel_12m_mensal/`.

Os eventos considerados sao homicidio doloso, latrocinio, tentativa de homicidio,
lesao corporal seguida de morte e feminicidio. A unidade geografica atual e o
municipio; a base nao permite conclusoes por bairro.

## Pipeline

1. `scripts/sanitizar_violencia_grave_sul.py`: filtra e agrega as planilhas anuais.
2. `scripts/criar_target_violencia_grave.py`: calcula o maximo historico anterior e o target.
3. `scripts/treinar_modelo_risco.py`: cria variaveis historicas e compara baseline,
   regressao logistica, arvore, Random Forest, Extra Trees, boosting e KNN.
4. `scripts/comparar_balanceamento.py`: compara dados originais, peso de classes e
   undersampling 1:3 e 1:2 nos principais algoritmos.
5. `scripts/otimizar_regressao_logistica.py`: compara regularizacao L1 e L2 com
   diferentes intensidades na regressao logistica com undersampling 1:5.
   Esse script pertence ao experimento anterior e foi mantido para registro.
6. `scripts/otimizar_gradient_boosting.py`: otimiza o Gradient Boosting escolhido
   para o alerta classificatorio dos proximos seis meses.
7. `scripts/validar_estabilidade_modelo.py`: agrega as previsoes fora do treino
   de 2020 a 2023, determina um limiar unico e verifica a variacao entre anos.
8. `scripts/analisar_limiares_modelo.py`: mede, somente nas validacoes, quanto a
   reducao de falsos alertas custa em casos reais nao identificados.
9. `scripts/comparar_perfis_municipios.py`: cria cinco perfis temporais de
   municipios e compara o Gradient Boosting com e sem essa contextualizacao.
10. `scripts/avaliar_modelo_final.py`: treina a configuracao fechada, executa uma
   avaliacao nas consultas de 2024-07 a 2025-06 e salva as previsoes e o
   artefato do modelo.

A comparacao faz validacao temporal progressiva nos anos de 2020 a 2023. Em cada
rodada, o treinamento usa apenas os anos anteriores. As consultas de 2024-07 a
2025-06 foram reservadas e acessadas uma unica vez no teste final. O limiar de
alerta foi definido nas validacoes pelo F2, que da mais peso ao recall, e o
ranking usou a media da `average precision` entre as quatro rodadas.

## Resultado final

O HistGradientBoosting `conservador_15`, sem balanceamento e com limiar de
0,1081, obteve no teste final AP de 0,2741, precisao de 0,2089, recall de 0,8772
e F2 de 0,5349. O limiar e os parametros nao foram reajustados depois da abertura
do teste.

## Execucao

```powershell
python -m pip install -r requirements.txt
python scripts/criar_target_violencia_grave.py
python scripts/treinar_modelo_risco.py
python scripts/comparar_balanceamento.py
python scripts/otimizar_gradient_boosting.py
python scripts/validar_estabilidade_modelo.py
python scripts/analisar_limiares_modelo.py
python scripts/comparar_perfis_municipios.py
python scripts/avaliar_modelo_final.py
python -m unittest discover -s tests -v
```

As metricas finais detalhadas ficam em
`reports/metricas_avaliacao_modelo_final.json`, e o resumo legivel em
`reports/resumo_avaliacao_modelo_final.txt`. As previsoes e o artefato treinado
tambem sao gerados pelo ultimo passo do pipeline.

## Limitacoes

- A ocorrencia de um novo maximo historico e rara e fica ainda mais rara com o
  aumento do periodo observado.
- A base usa volume absoluto de vitimas, sem ajuste pela populacao municipal.
- O alerta deve ser combinado com estudos de seguranca, mercado, infraestrutura
  e viabilidade antes de qualquer decisao empresarial.
