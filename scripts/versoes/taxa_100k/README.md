# Versao atual: taxa por 100 mil habitantes

Esta e a versao escolhida para continuar a aplicacao end-to-end.

## Regra de classificacao

O target vale `1` quando a taxa acumulada dos proximos seis meses supera 25 por
100 mil habitantes e ha ocorrencias em pelo menos dois meses diferentes. Caso
contrario, vale `0`.

## Ordem dos scripts

1. `criar_target_taxa_100k.py`: gera o target proporcional.
2. `treinar_modelo_taxa_100k.py`: compara os algoritmos.
3. `otimizar_modelo_taxa_100k.py`: otimiza Random Forest e Gradient Boosting.
4. `empacotar_modelo.py`: reajusta o vencedor com todos os targets conhecidos e
   salva o modelo, o limiar e os metadados para a API.

Execute os arquivos como modulos a partir da raiz do projeto:

```powershell
python -m scripts.versoes.taxa_100k.criar_target_taxa_100k
python -m scripts.versoes.taxa_100k.treinar_modelo_taxa_100k
python -m scripts.versoes.taxa_100k.otimizar_modelo_taxa_100k
python -m scripts.versoes.taxa_100k.empacotar_modelo
```

O modelo selecionado foi a Random Forest com 300 arvores, profundidade maxima
14, minimo de 10 exemplos por folha e limiar 0,1313.
