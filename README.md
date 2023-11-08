### Часть 1:
- реализовать класс набора данных OxfordIIITPet
- реализовать вычисление метрик accuracy, balanced accuracy
- реализовать класс обучения Trainer (сохранение и выгрузку весов модели, логирование значений целевой функции и метрик на каждом шаге обучения, реализовать функции обучения на train set и evaluation на test set, обучение проводить на gpu)
- реализовать логирование в neptune.ai в классе Trainer
- подобрать препроцессинг данных на основе следующих преобразований: 
    - Обучающие данные:
      1. RandomResizedCrop(224); 
      2. RandomHorizontalFlip(). 
      3. ColorJitter: scale hue, saturation, and brightness with coefficients uniformly drawn from [0.6, 1.4].
      4. Normalize(mean=[0.485, 0.456, 0.405], std=[0.229, 0.224, 0.225]) 

    - Тестовые данные:
      1. Resize(256)
      2. CenterCrop(224)
      3. Normalize(mean=[0.485, 0.456, 0.405], std=[0.229, 0.224, 0.225])
- визуализация батча до и после применения аугментаций (функция show_batch)

### Часть 2:
- реализовать функцию conv_block (models/blocks/vgg16_blocks.py)
- реализовать функцию classifier_block (models/blocks/vgg16_blocks.py)
- реализовать нейронную сеть VGG-16, используя функции conv_block и classifier_block
- обучить VGG-16 на наборе данных OxfordIIITPet

![VGG-16](https://neurohive.io/wp-content/uploads/2018/11/vgg16-neural-network-1.jpg)

### Часть 3:
- реализовать классы блоков для ResNet-50 (models/blocks/resnet_blocks.py)
- реализовать нейронную сеть ResNet-50, используя предыдущий пункт
- добавить в ResNet-50 инициализацию весов
- в классе Trainer добавить возможность выбрать модель для обучения согласно cfg.model_name в train_cfg.py
- обучить базовую модель ResNet-50 на наборе данных OxfordIIITPet


### Общие требования к обучению моделей:
1. Визуализация батча до и после применения аугментаций (utils/visualization.py функция show_batch)
2. Каждый эксперимент и его параметры должны быть залогированы в neptune.ai (с понятным название эксперимента)
3. Во время обучения логировать значение целевой функции, accuracy, balanced accuracy на обучающей выборке на каждом шаге
4. После окончания каждой эпохи обучения посчитать значения целевой функции, accuracy, balanced accuracy на тестовых данных, полученные значения логировать в neptune.ai
5. Сохранить модель с лучшим значением метрик на тестовой выборке (класс Trainer метод save_model)
6. Все модели обучать до насыщения accuracy на тестовой выборке
7. Обучать на GPU
8. Во время обучения переводить модель в режим model.train(), во время evaluation - model.eval()


### Полезные ссылки
- https://www.robots.ox.ac.uk/~vgg/data/pets/
- https://www.kaggle.com/datasets/polinakhanzhina/oxford-iiit-pet
- https://arxiv.org/pdf/1409.1556.pdf
- https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- https://github.com/pkhanzhina/mllib_f2023_mlp/blob/master/executors/mlp_trainer.py
- https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html
- https://pytorch.org/vision/main/auto_examples/others/plot_visualization_utils.html#sphx-glr-auto-examples-others-plot-visualization-utils-py
- https://arxiv.org/pdf/1512.03385.pdf
- https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.pdf
