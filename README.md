# AntiVlad

AlphaZero-подобный шахматный агент.

## Архитектура

```text
constants.py       # константы и таблицы кодирования действий
encode_state.py    # история позиций -> state tensor
action_encoder.py  # move <-> action_id, legal mask
antivlad_net.py    # policy + value network
mcts.py            # MCTS
self_play.py       # self-play и сбор данных
dataset.py         # PyTorch Dataset
train.py           # обучение
```

## Состояние

Размер входа:

```text
(101, 8, 8)
```

История содержит **8 позиций**, от старой к текущей:

```text
history[0] = t-7
...
history[7] = t
```

Если позиций меньше 8, самая ранняя позиция повторяется.

### Плоскости

Каждая позиция содержит 12 плоскостей:

```text
0-5   свои:    pawn knight bishop rook queen king
6-11  чужие:   pawn knight bishop rook queen king
```

8 позиций:

```text
8 × 12 = 96
```

Дополнительные плоскости текущей позиции:

```text
96-99   права на рокировку:
        своя короткая
        своя длинная
        чужая короткая
        чужая длинная

100     поле en passant
```

Итого:

```text
96 + 4 + 1 = 101
```

### Перспектива

`own/opponent` определяется относительно игрока, который ходит сейчас:

```python
current_player = history[-1].turn
```

Эта перспектива используется для всех 8 позиций.

Геометрия доски фиксирована. Доска не поворачивается и не отражается для чёрных.

```text
a1 -> одна и та же ячейка tensor
h8 -> одна и та же ячейка tensor
```

### История

`encode_state()` получает готовую историю из 8 позиций.

Заполнение истории выполняется в `self_play.py`.

## Действия

```text
ACTION_SIZE = 8 * 8 * 73
            = 4672
```

73 типа действий:

```text
0-55   дальние ходы
56-63  ходы конём
64-72  превращения пешки
```

Кодирование и декодирование действий находятся в `action_encoder.py`.

## MCTS

```text
Selection
→ Expansion
→ Evaluation
→ Backup
```

Нейросеть выдаёт:

```text
policy logits
value
```

При `backup` знак `value` меняется на каждом уровне дерева.

## Правила

* `python-chess` отвечает за правила шахмат.
* `encode_state.py` отвечает за представление состояния.
* `action_encoder.py` отвечает за действия.
* `antivlad_net.py` не знает о MCTS и правилах шахмат.
* `dataset.py` не знает о шахматах и MCTS.
* Фиксированные параметры находятся в `constants.py`.
