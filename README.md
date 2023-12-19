Репозиторий называется `generating-pokemons`, но научиться генерировать покемонов я не смог,
поэтому предлагаю посмотреть на котиков.

## Preliminaries

```commandline
pip install -r requirements.txt
```

## Запуск обучения

```commandline
python3 train.py
```

[Датасет с мордочками котов](https://www.kaggle.com/datasets/spandan2/cats-faces-64x64-for-generative-models) будет загружен автоматически.

Можно указать следующие аргументы:
- `-i/--input`: путь до директории с изображениями
- `-q`: флаг, который отключает логирование в wandb

## Evaluation


