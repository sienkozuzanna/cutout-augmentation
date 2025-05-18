# Results of experiments

## Dataset: CIFAIR

| Cutout Type | Max Size / Param   | Soft / Normal | Val Acc | Train Acc |
|-------------|--------------------|----------------|---------|-----------|
| Polygon     | 0.2                | Normal         | 0.8033  | 0.8461    |
| Polygon     | 0.1                | Normal         | 0.8023  | 0.8661    |
| Polygon     | 0.05               | Normal         | 0.7989  | 0.8687    |
| Polygon     | 0.05               | Soft           | 0.7833  | 0.8705    |
| Pixels      | 0.1                | Normal         | 0.7709  | 0.8715    |
| Pixels      | 0.05               | Normal         | 0.8010  | 0.8619    |
| Pixels      | 0.05               | Soft           | 0.8123  | 0.8656    |
| Pixels      | 0.05               | Soft           | 0.4791  | 0.5017    |
| Squares     | 25, ratio = -0.1   | Normal         | 0.7601  | 0.8584    |
| Squares     | 10, ratio = 0.1    | Normal         | 0.8055  | 0.8627    |
| Squares     | 10, ratio = 0.05   | Normal         | 0.8135  | 0.8637    |
| Squares     | 10, ratio = 0.05   | Soft           | 0.7912  | 0.8646    |
| Square      | 5                  | Normal         | 0.8119  | 0.8747    |
| Square      | 10                 | Normal         | 0.8010  | 0.8633    |
| Square      | 5                  | Soft           | 0.7460  | 0.8117    |

---

## Dataset: Fashion MNIST

| Cutout Type | Max Size / Param   | Soft / Normal | Val Acc | Train Acc |
|-------------|--------------------|----------------|---------|-----------|
| Polygon     | 0.2                | Normal         | 0.9276  | 0.9446    |
| Polygon     | 0.1                | Normal         | 0.9238  | 0.9464    |
| Polygon     | 0.05               | Normal         | 0.9274  | 0.9461    |
| Polygon     | 0.05               | Soft           | 0.9064  | 0.9472    |
| Pixels      | 0.1                | Normal         | 0.9193  | 0.9473    |
| Pixels      | 0.05               | Normal         | 0.9244  | 0.9471    |
| Pixels      | 0.05               | Soft           | 0.9265  | 0.9481    |
| Pixels      | 0.05               | Soft           | 0.7149  | 0.6800    |
| Squares     | 25, ratio = 0.1    | Normal         | 0.9200  | 0.9451    |
| Squares     | 10, ratio = 0.1    | Normal         | 0.9269  | 0.9476    |
| Squares     | 10, ratio = 0.05   | Normal         | 0.7253  | 0.8099    |
| Squares     | 10, ratio = 0.05   | Soft           | 0.7701  | 0.7879    |
| Square      | 5                  | Normal         | 0.9262  | 0.9471    |
| Square      | 10                 | Normal         | 0.9211  | 0.9475    |
| Square      | 5                  | Soft           | 0.8063  | 0.7243    |
