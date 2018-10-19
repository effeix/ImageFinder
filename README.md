# ImageFinder

Este programa tem como objetivo implementar um algoritmo para busca de imagens utilizando outras imagens semelhantes. O funcionamento do programa não é extremanente complexo, podendo ser reproduzido facilmente. Os programa consiste em duas partes: treinamento e busca.

Na parte de treinamento, a tarefa é criar um banco de imagens representadas por seus respectivos histogramas. Primeiramente, são extraídos descritores de cada imagem (utilizando o algoritmo SURF do OpenCV). Estes descritores serão utilizados para criar um vocabulário. Este vocabulário é criado utilizando um método de clusterização, o KMeans. Este método consiste em criar grupos (clusters) distintos de descritores, que serão utilizados para a criação dos histogramas, a última etapa. O histograma representa a frequência de um cluster no conjunto de histogramas (quantas vezes um cluster foi populado por um descritor).

Neste momento temos um conjunto de imagens representados por histogramas. Assim, para realizarmos a busca, coletamos a imagem a ser buscada, a transformamos em uma representação por histograma e comparamos com os histogramas do banco (montados na etapa de treinamento). Os histogramas mais semelhantes ao da imagem buscada são os resultados corretos.

# Utilização

No momento não há uma distinção entre etapas de treinamento e busca, as duas etapas são realizadas de uma vez quando o usuário executar o programa. Da mesma maneira, ainda não é possível escolher quais imagens serão utilizadas para treinamento sem uma alteração direta no código. Para executar, utilize:

```sh
$ python search.py <caminho_imagem>
```

onde *caminho_imagem* é o caminho do arquivo contendo a imagem a ser buscada. O programa imprimirá o caminho das 5 imagens mais semelhantes encontradas.

# To Do
- Separar treinamento e busca (salvar treinamento)
- Permitir treinamento customizado
- Permitir escolha do método de similaridade
- Persistir imagens e histogramas
- Webservice
