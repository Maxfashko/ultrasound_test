# Ultrasound Nerve Segmentation

Identify nerve structures in ultrasound images of the neck

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

1) you will need to install a requirements to run the `pip install -r requirements.txt`

2) change the data path `utils/params.py`

## Running

1) create train, test data

```
python bin/create_data.py
```

2) run training

```
python bin/train.py
```

3) run submission

```
python bin/submit.py
```

# Results

- Best single model achieved 0.68793 LB score
