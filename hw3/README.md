# dsp-hw

## Третье задание. Использование фильтра


Прогнал через DeepFilterNet и потом тем же скриптом получил часть метрик, остальные вручную.

|    | Filename                  | SNR (before filtering) |      SDR |   SI-SDR |    PESQ | NISQA   | DNSMOS   | MOS   |
|---:|:--------------------------|------:|---------:|---------:|--------:|:--------|:---------|:------|
|  0 | mix_-5_DeepFilterNet3.wav |    -5 |  3.39249 | -45.3224 | 1.23607 | 3.045371  3.017256  3.967818  3.427338 | 3.3585193 | 2.5 (В начале почти идеально, в конце такое чувство что фильтр только хуже сделал) | 
|  1 | mix_0_DeepFilterNet3.wav  |     0 |  8.27664 | -45.0861 | 1.54594 | 3.774391  3.745236  4.243632  3.983888 | 3.989788 | 3.75 |
|  2 | mix_5_DeepFilterNet3.wav  |     5 | 12.2845  | -44.9898 | 2.1792  | 4.209193  4.132720  4.430441  4.110410 | 4.0497723 | 4.5 |
|  3 | mix_10_DeepFilterNet3.wav |    10 | 15.8023  | -45.015  | 2.68805 | 4.404206  4.225575  4.498682  4.185361 | 4.164157 | 4.75 |

