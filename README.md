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

## Inference

Можно сгенерировать изображения обученной моделью с помощью скрипта
`inference.py`.

Он поддерживает 2 вида генерации:
- несколько изображений в виде отдельных файлов

```commandline
python3 inference.py {кол-во изображений}
```
например: `python3 inference.py 5` сгенерирует файлы `000.png`, ..., `004.png`

- несколько изображений сеткой в виде одного файла 
```commandline
python3 inference.py -g {кол-во строк сетки} {кол-во столбцов сетки}
```
например: `python3 inference.py -g 3 5` сгенерирует файл `grid.png` с сеткой размера 3x5

Веса генератора будут загружены автоматически.
Если по каким-то причинам ссылки перестали работать (на момент написания они работают), то
чекпоинты можно загрузить вручную и затем поместить в директорию `inference/checkpoints/{arch_name}`:
- [gan_big](https://drive.google.com/file/d/1nFUBPYrKDO0_VTF1qRFu3_ApHy8hx9zX/view?usp=drive_link) -> `inference/checkpoints/gan_big/generator_checkpoint.pth`

Скрипт поддерживает несколько аргументов (можно посмотреть вызовом `python3 inference.py --help`):
- `-s/--storage`: путь до директории с чекпоинтами генераторов, чекпоинты должны в ней находиться
по путям `{arch_name}/generator_checkpoint.pth` (по умолчанию - `inference/checkpoints`)
- `-o/--output`: путь до директории, куда сохранять сгенерированные изображения (по умолчанию - `inference/generated`)
- `-m/--model`: имя архитектуры используемого генератора (по умолчанию - `gan_big`)
- `-b/--batch`: размер батча для генератора (по умолчанию - 32)
