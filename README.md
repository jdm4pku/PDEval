# PDEval: An Evaluation of Requirements Modeling for Cyber-Physical System via LLMs

## Table of Contents

- Requirements
- Datasets
- Usage



## Requirements

* vllm
* openai

## Dataset

The dataset can be access at [data directory in this project](https://github.com/publicsubmission/CPSBench/tree/main/data/dataset/10-fold) .

Here is an sample in this dataset. 

```
{
    "text": "A DigitalHome System shall have the capability to establish an individual home web server hosted on a home computer.",
    "entity": {
      "Machine Domain": [
        "A DigitalHome System"
      ],
      "Physical Device": [
        "a home computer"
      ],
      "Environment Entity": [],
      "Design Domain": [
        "an individual home web server"
      ],
      "Requirements": [],
      "Shared Phenomena": []
    },
    "relation": {
      "interface": [
        [
          "A DigitalHome System",
          "an individual home web server"
        ],
        [
          "a home computer",
          "an individual home web server"
        ]
      ],
      "requirements reference": [],
      "requirements constraints": []
    }
 }
```

## Usage

### Setup

* git clone https://github.com/publicsubmission/CPSBench.git
* conda create -n CPSBech python=3.9
* conda activate CPSBench
* pip install vllm, pip install openai

### Evaluation

* bash ./script/ner.sh

* bash ./script/rel_llm.sh



