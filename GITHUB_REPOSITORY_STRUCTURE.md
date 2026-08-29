# GitHub repository structure

```text
sensor-faithfulness-rsvqa/
├── README.md
├── LICENSE
├── CITATION.cff
├── THIRD_PARTY_NOTICES.md
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── configs/
│   ├── models.yaml
│   ├── datasets.yaml
│   ├── m4sar_protocol.yaml
│   ├── bigearthnet_protocol.yaml
│   └── bigearthnet_rendering.json
├── src/sensor_faithfulness/
│   ├── data/
│   ├── inference/
│   ├── evaluation/
│   ├── paper/
│   └── utils/
├── scripts/
├── fixtures/
│   ├── m4sar/
│   └── bigearthnet/
├── metadata/
├── examples/small_demo/
├── results/paper_summary/
├── figures/
├── tests/
└── docs/
```

The release repository is independent from the full research workspace. It
contains only reproduction-critical source, configuration, redistributable
metadata, aggregate paper results, and generated paper artifacts.
